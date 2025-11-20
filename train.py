from src.utils.init_utils import set_random_seed, init_logger_saving_resume, get_accelerator, get_metrics, get_param_groups
from src.trainer import Trainer
from src.datasets.data_utils import get_dataloaders
from src.logger import ModelLoader
from omegaconf import OmegaConf
from hydra.utils import instantiate
import torch
import hydra
import warnings

from src.logger import DummyWriter
from dotenv import load_dotenv
load_dotenv()


warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    accelerator = get_accelerator()

    device = accelerator.device
    is_main = accelerator.is_main_process

    set_random_seed(config.trainer.seed + accelerator.process_index)

    # config = CometMLWriter.download_model_checkpoint(config, logger=None)  # 1!!!
    resume_from_checkpoint = config.trainer.get("resume_from", None) is not None
    logger, save_dir, resume_path, config = init_logger_saving_resume(config, accelerator)
    logger.info(f"SAVE DIR : {save_dir}")

    # setup_saving_and_logging(config, is_main)
    project_config = OmegaConf.to_container(config)

    writer = DummyWriter()
    if is_main:
        writer = instantiate(config.writer, logger, project_config, resume=resume_from_checkpoint)

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device, accelerator)

    model_loader = ModelLoader(config, logger, accelerator)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(
        config.loss_function).to(device)

    metrics = get_metrics(config)

    # build optimizer, learning rate scheduler
    param_groups = get_param_groups(config, model)

    optimizer = instantiate(config.optimizer, params=param_groups, _convert_="object")

    epoch_len = config.trainer.get("epoch_len")
    if epoch_len is not None:
        epoch_len = epoch_len // accelerator.num_processes
    else:
        config.trainer["epoch_len"] = len(dataloaders["train"])
        print(f"Set epoch_len to {config.trainer['epoch_len']}")

    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer, _convert_="object")

    if resume_from_checkpoint is not None:
        model = model_loader.load(model, save_dir)

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    trainer = Trainer(
        model=model,
        accelerator=accelerator,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
        use_profiler=config.trainer.get("use_profiler", False),
        checkpoint_dir=save_dir,
        resume_from=resume_path,
    )

    trainer.train()


if __name__ == "__main__":
    main()
