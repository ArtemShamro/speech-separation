import logging
import os
import random
import secrets
import shutil
import string
import subprocess

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import broadcast_object_list
import numpy as np
import torch
from omegaconf import OmegaConf
from datetime import datetime
from hydra.utils import instantiate, get_class
from pathlib import Path

from src.logger.logger import setup_logging
from src.utils.io_utils import ROOT_PATH


def get_param_groups(config, model):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "LayerNorm.weight" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {
            "params": decay_params,
            "weight_decay": config.optimizer.get("weight_decay", 0.0),
        },
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return param_groups


def get_accelerator() -> Accelerator:
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(non_blocking=True),
    )

    if accelerator.state.dynamo_plugin is not None:
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.optimize_ddp = False
        torch._dynamo.config.suppress_errors = True
        # torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

    return accelerator


def get_metrics(config):
    metrics = {"train": [], "inference": []}
    for metric_type in ["train", "inference"]:
        for metric_config in config.metrics.get(metric_type, []):
            # use text_encoder in metrics
            metrics[metric_type].append(instantiate(metric_config))
    return metrics


def set_worker_seed(worker_id):
    """
    Set seed for each dataloader worker.

    For more info, see https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        worker_id (int): id of the worker.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_random_seed(seed):
    """
    Set random seed for model training or inference.

    Args:
        seed (int): defines which seed to use.
    """
    # fix random seeds for reproducibility
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=True works faster but reproducibility decreases
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# https://github.com/wandb/wandb/blob/main/wandb/sdk/lib/runid.py
def generate_id(length: int = 8) -> str:
    """
    Generate a random base-36 string of `length` digits.

    Args:
        length (int): length of a string.
    Returns:
        run_id (str): base-36 string with an experiment id.
    """
    # There are ~2.8T base-36 8-digit strings. If we generate 210k ids,
    # we'll have a ~1% chance of collision.
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def log_git_commit_and_patch(save_dir):
    """
    Log current git commit and patch to save dir.
    Improves reproducibility by allowing to run the same code version:
        git checkout commit_hash_from_commit_path
        git apply patch_path

    If you created new files and want to have them in patch,
    stage them via git add before running the script.

    Patch can be applied via the following command:
        git apply patch_path

    Args:
        save_dir (Path): directory to save patch and commit in
    """
    print("Logging git commit and patch...")
    commit_path = save_dir / "git_commit.txt"
    patch_path = save_dir / "git_diff.patch"
    with commit_path.open("w") as f:
        subprocess.call(["git", "rev-parse", "HEAD"], stdout=f)
    with patch_path.open("w") as f:
        subprocess.call(["git", "diff", "HEAD"], stdout=f)


def saving_init(save_dir, config):
    """
    Prepare save directory and set run_id.

    Args:
        save_dir (Path): directory for logs, checkpoints, config, etc.
        config (DictConfig): Hydra config.
    """
    if save_dir.exists():
        if config.trainer.override:
            print(f"Overriding save directory '{save_dir}'...")
            shutil.rmtree(str(save_dir))
        else:
            raise ValueError(
                f"Save directory '{save_dir}' exists. "
                "Change run_name or set trainer.override=True."
            )

    save_dir.mkdir(exist_ok=True, parents=True)

    # генерируем новый run_id (всегда при новом запуске)
    run_id = generate_id(length=config.writer.id_length)

    OmegaConf.set_struct(config, False)
    config.writer.run_id = run_id
    OmegaConf.set_struct(config, True)

    return save_dir


def resume_from_checkpoint_init(config, logger, save_dir, accelerator: Accelerator):
    """
    Resume logic: either from local checkpoint (resume_from path),
    or from remote (writer-specific download method).
    Returns updated config.
    """
    resume_path = config.trainer.get("resume_from")
    if resume_path is None:
        return config, None

    if accelerator.is_main_process:
        if not Path(resume_path).exists():
            # резюм через writer-класс (например, CometML)
            WriterClass = get_class(config.writer._target_)
            # предполагаем, что класс умеет вернуть (config, checkpoint_path)
            resume_path = WriterClass.download_model_checkpoint(
                config, resume_path, save_dir, logger
            )

        # broadcast
        obj = [str(resume_path)]
    else:
        obj = [None]
    accelerator.wait_for_everyone()
    resume_path = Path(broadcast_object_list(obj, from_process=0)[0])  # type: ignore
    checkpoint = torch.load(resume_path, map_location="cpu")

    if "config" in checkpoint:
        resume_config = checkpoint["config"]
    else:
        raise ValueError(f"No config found in checkpoint {resume_path}")

    if accelerator.is_main_process:
        OmegaConf.save(resume_config, save_dir / "config.yaml")
        log_git_commit_and_patch(save_dir)

    return resume_config, resume_path


def init_logger_saving_resume(config, accelerator: Accelerator):
    """
    Initialize logger and save directory.

    Args:
        config (DictConfig): Hydra config.
        is_main (bool): whether this is the main process.
    Returns:
        logger, save_dir
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = (
        ROOT_PATH / config.trainer.save_dir / f"{config.writer.run_name}_{timestamp}"
    )

    if not accelerator.is_main_process:
        logger = logging.getLogger("train-dummy")
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
    else:
        saving_init(save_dir, config)
        if config.trainer.get("resume_from") is not None:
            setup_logging(save_dir, append=True)
        else:
            setup_logging(save_dir, append=False)

        logger = logging.getLogger("train")
        logger.setLevel(logging.DEBUG)

    # применяем resume-логику
    config, resume_path = resume_from_checkpoint_init(
        config, logger, save_dir, accelerator
    )

    return logger, save_dir, resume_path, config
