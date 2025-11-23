import os
import torch
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from src.logger.utils import align_state_dict_keys
from hydra.utils import get_class
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list


class ModelLoader:
    def __init__(self, config, logger, accelerator: Accelerator | None = None):
        """
        Initialize the ModelLoader instance.

        Args:
            config (DictConfig | dict): Hydra config or a plain dictionary with model loader parameters.
            logger (logging.Logger): Logger instance for reporting progress and issues.
            accelerator (Accelerator, optional): Accelerator object for distributed or mixed precision inference.
        """
        self.project_config = config
        self.config = config.model_loader
        self.logger = logger
        self.accelerator = accelerator

    def load(self, model, save_dir):
        """
        Load pretrained model weights from local storage, CometML, or Hugging Face Hub.

        Args:
            model (torch.nn.Module): The model instance to load weights into.
            save_dir (str | Path): Directory to save or temporarily store downloaded checkpoints.

        Returns:
            torch.nn.Module: Model with loaded pretrained weights.
        """
        if self.config.get("from_pretrained", "no") != "no":
            path_to_pretrained = self.config.get("from_pretrained")
            if str(path_to_pretrained).startswith("writer:"):
                if self.accelerator is None or self.accelerator.is_main_process:
                    experiment_id = path_to_pretrained.replace("writer:", "")
                    self.logger.info(
                        f"Loading pretrained model from CometML experiment {experiment_id}"
                    )
                    WriterClass = get_class(self.project_config.writer._target_)
                    path_to_pretrained = WriterClass.download_model_checkpoint(
                        self.project_config, experiment_id, save_dir, self.logger
                    )
                    obj = [str(path_to_pretrained)]
                else:
                    obj = [None]
                if self.accelerator is not None:
                    self.accelerator.wait_for_everyone()
                path_to_pretrained = Path(
                    broadcast_object_list(obj, from_process=0)[0]
                )  # type: ignore
            if Path(path_to_pretrained).exists():
                self.logger.info(f"Loading local checkpoint: {path_to_pretrained}")
                return self._from_local_pretrained(model, path_to_pretrained)
            else:
                self.logger.info(
                    f"Loading pretrained model from Hugging Face: {path_to_pretrained}"
                )
                return self._from_hf_pretrained(model, path_to_pretrained)
        return model

    @classmethod
    def push_to_hugging_face(cls, repo_and_branch, checkpoint_dir, writer_id, logger):
        """
        Upload a trained model checkpoint to the Hugging Face Hub.

        Args:
            repo_and_branch (str): Combined string with repository ID and branch (e.g., "user/repo@branch").
            checkpoint_dir (str | Path): Directory containing the model checkpoint.
            writer_id (str): Identifier of the CometML experiment for traceability.
            logger (logging.Logger): Logger instance for progress tracking.

        Returns:
            tuple | None: (repo_id, branch, commit_url) if successful, otherwise None.
        """
        token = os.getenv("HF_TOKEN")
        api = HfApi()
        repo_id, branch = repo_and_branch.split("@")

        if branch != "main":
            try:
                api.create_branch(
                    repo_id=repo_id, branch=branch, repo_type="model", token=token
                )
                logger.info(f"Created or verified branch: {branch}")
            except Exception as e:
                msg = str(e)
                if "already exists" in msg or "409" in msg:
                    logger.info(f"Branch '{branch}' already exists, skipping creation.")
                else:
                    logger.warning(f"Could not create branch '{branch}': {msg}")

        full_checkpoint_path = Path(checkpoint_dir) / "model_best.pth"

        if not full_checkpoint_path.exists():
            logger.warning(f"No model_best.pth found at {full_checkpoint_path}")
            return

        try:
            checkpoint = torch.load(
                full_checkpoint_path, map_location="cpu", weights_only=False
            )
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = align_state_dict_keys(checkpoint["state_dict"])
            else:
                state_dict = align_state_dict_keys(checkpoint)
        except Exception as e:
            logger.error(f"Failed to load checkpoint at {full_checkpoint_path}: {e}")
            return

        weights_path = Path(checkpoint_dir) / "model_weights.pth"
        try:
            torch.save(state_dict, weights_path)
            logger.info(f"Saved lightweight model weights to {weights_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return

        logger.info(f"Uploading model to Hugging Face Hub: {repo_id}@{branch}")

        try:
            info = api.upload_file(
                path_or_fileobj=str(weights_path),
                path_in_repo="model.pth",
                repo_id=repo_id,
                repo_type="model",
                revision=branch,
                commit_message=f"Comet exp id: {writer_id}",
                token=token,
            )

            commit_url = getattr(info, "commit_url", None)
            logger.info(f"Model weights uploaded successfully. Commit id: {commit_url}")

            return (repo_id, branch, commit_url)

        except Exception as e:
            logger.error(f"Failed to push model to Hugging Face: {e}")

    def _from_hf_pretrained(self, model, path_to_pretrained):
        """
        Load model weights from the Hugging Face Hub.

        Args:
            model (torch.nn.Module): Model instance to load weights into.
            path_to_pretrained (str): Hugging Face repository and branch in the format "repo_id@revision".

        Returns:
            torch.nn.Module: Model with loaded weights.
        """
        filename = "model.pth"
        repo_id, revision = path_to_pretrained.split("@")
        try:
            self.logger.info(
                f"Downloading model from Hugging Face Hub: {repo_id}@{revision} ({filename})"
            )

            model_path = hf_hub_download(
                repo_id=repo_id, filename="model.pth", revision=revision
            )
            device = next(model.parameters()).device
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)

            model = self._load_checkpoint(model, checkpoint)

            self.logger.info(
                f"Model weights successfully loaded from Hugging Face: {repo_id}@{revision}"
            )

        except Exception as e:
            self.logger.error(f"Failed to load model from Hugging Face: {e}")
            raise e

        return model

    def _from_local_pretrained(self, model, path_to_pretrained):
        """
        Load model weights from a local checkpoint file.

        Args:
            model (torch.nn.Module): Model instance to load weights into.
            path_to_pretrained (str | Path): Path to the local checkpoint file.

        Returns:
            torch.nn.Module: Model with loaded weights.
        """
        pretrained_path = str(path_to_pretrained)
        self.logger.info(f"Loading model weights from: {pretrained_path} ...")

        device = next(model.parameters()).device
        checkpoint = torch.load(
            pretrained_path, map_location=device, weights_only=False
        )

        model = self._load_checkpoint(model, checkpoint)

        self.logger.info(f"Model weights successfully loaded from {pretrained_path}")
        return model

    def _load_checkpoint(self, model, checkpoint):
        """
        Load a checkpoint dictionary into the model.

        Args:
            model (torch.nn.Module): The model to update with checkpoint weights.
            checkpoint (dict | OrderedDict): The checkpoint data, potentially containing a 'state_dict' key.

        Returns:
            torch.nn.Module: Model with updated parameters.
        """
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = align_state_dict_keys(checkpoint["state_dict"])
        else:
            state_dict = align_state_dict_keys(checkpoint)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if missing:
            self.logger.warning(f"Missing keys when loading state_dict: {missing}")
        if unexpected:
            self.logger.warning(f"Unexpected keys in state_dict: {unexpected}")

        return model
