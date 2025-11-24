from abc import abstractmethod

import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.logger import ModelLoader
from src.logger.utils import align_state_dict_keys
from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH

from accelerate import Accelerator, utils
import contextlib
import random


class BaseTrainer:
    """
    Base class for all trainers.
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        config,
        device,
        dataloaders,
        logger,
        writer,
        accelerator=None,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
        use_profiler=False,
        train_log_items=10,
        checkpoint_dir=None,
        resume_from=None,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the
                optimizer.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        self.is_train = True
        self.accelerator: Accelerator | None = accelerator
        self.is_main_process = self.accelerator is None or self.accelerator.is_main_process
        self.use_profiler = use_profiler

        self.train_log_items = train_log_items

        self.config = config
        self.cfg_trainer = self.config.trainer

        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = config.trainer.get("log_step", None)

        self.model: torch.nn.Module = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.train_dataloader = dataloaders["train"]
        if self.log_step is None:
            # epoch-based logging
            self.log_step = len(self.train_dataloader)
        if epoch_len is None:
            # epoch-based training
            self.epoch_len = len(self.train_dataloader)
        else:
            # iteration-based training
            self.epoch_len = epoch_len
            self._base_train_dataloader = self.train_dataloader
            if self.epoch_len > len(self.train_dataloader):
                self.train_dataloader = inf_loop(self.train_dataloader)

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        # configuration to monitor model performance and save best

        self.save_period = (
            self.cfg_trainer.save_period
        )  # checkpoint each save_period epochs
        self.monitor = self.cfg_trainer.get(
            "monitor", "off"
        )  # format: "mnt_mode mnt_metric"

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        # setup visualization writer instance
        self.writer = writer

        # define metrics
        self.metrics = metrics
        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            "grad_norm",
            *[m.name for m in self.metrics["train"]],
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["inference"]],
        )

        # define checkpoint dir and init everything if required
        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir is None:
            self.checkpoint_dir = (
                ROOT_PATH / config.trainer.save_dir / config.writer.run_name
            )

        if resume_from is not None:
            self._resume_checkpoint(resume_from)

    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic:

        Training model for an epoch, evaluating it on non-train partitions,
        and monitoring the performance improvement (for early stopping
        and saving the best checkpoint).
        """
        try:
            not_improved_count = 0
            for epoch in range(self.start_epoch, self.epochs + 1):
                self._last_epoch = epoch

                if hasattr(self, "_base_train_dataloader") and hasattr(self._base_train_dataloader, "scheduled_transforms"):
                    self._base_train_dataloader.dataset.epoch_set(epoch)
                    if self.epoch_len > len(self.train_dataloader):
                        self.train_dataloader = inf_loop(self.train_dataloader)
                    self.train_dataloader = inf_loop(self._base_train_dataloader)

                result = self._train_epoch(epoch)

                # save logged information into logs dict
                logs = {"epoch": epoch}
                logs.update(result)

                # print logged information to the screen
                for key, value in logs.items():
                    self.logger.info(f"    {key:15s}: {value}")

                # evaluate model performance according to configured metric,
                # save best checkpoint as model_best
                best, stop_process = False, False
                if self.is_main_process:
                    best, stop_process, not_improved_count = self._monitor_performance(
                        logs, not_improved_count
                    )
                    if epoch % self.save_period == 0 or best:
                        self._save_checkpoint(epoch, save_best=best, only_best=False)

                if self.accelerator is not None:
                    self.accelerator.wait_for_everyone()
                    stop_tensor = torch.tensor(int(stop_process), device=self.accelerator.device)
                    self.accelerator.wait_for_everyone()
                    utils.broadcast(stop_tensor, from_process=0)
                    stop_process = bool(stop_tensor.item())

                if stop_process:
                    break

        finally:
            try:
                self.accelerator.wait_for_everyone()
            except Exception:
                pass
            try:
                self.accelerator.end_training()
            except Exception:
                pass

        if self.is_main_process and self.config.trainer.get("hugging_face_repo", False):
            self._push_to_hugging_face()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch, including logging and evaluation on
        non-train partitions.

        Args:
            epoch (int): current training epoch.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)

        if self.use_profiler:
            self.logger.info("Profiler enabled")
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=3, warmup=3, active=1, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    "./profiler_logs"),
                record_shapes=True,
                with_stack=True
            )
            profiler_ctx = profiler
        else:
            profiler_ctx = contextlib.nullcontext()
    
        with profiler_ctx as prof:
            for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train",
                     total=self.epoch_len, disable=(not self.is_main_process))
            ):
                batch['epoch'] = epoch
                try:
                    batch = self.process_batch(
                        batch,
                        metrics=self.train_metrics,
                    )
                except torch.cuda.OutOfMemoryError as e:
                    if self.skip_oom:
                        self.logger.warning("OOM on batch. Skipping batch.")
                        torch.cuda.empty_cache()  # free some memory
                        continue
                    else:
                        raise e
                if self.use_profiler:
                    try:
                        prof.step()
                    except RuntimeError:
                        pass

                self.train_metrics.update("grad_norm", self._get_grad_norm())

                # log current results
                if batch_idx % self.log_step == 0:
                    self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                    self.logger.debug(
                        "Train Epoch: {} {} Loss: {:.6f}".format(
                            epoch, self._progress(batch_idx), batch["loss"].item()
                        )
                    )
                    self.writer.add_scalar(
                        "learning rate", self.lr_scheduler.get_last_lr()[0]
                    )
                    self._log_scalars(self.train_metrics)
                    self._log_batch(batch_idx, batch)
                    self._log_audio(batch_idx, batch, self.train_log_items)
                    # we don't want to reset train metrics at the start of every epoch
                    # because we are interested in recent train metrics
                    last_train_metrics = self.train_metrics.result()
                    self.train_metrics.reset()

                if batch_idx >= self.epoch_len:
                    break

        logs = last_train_metrics

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        val_logs_total = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)

            if self.accelerator is not None:
                reduced = {}
                for k, v in val_logs.items():
                    t = torch.tensor([float(v)], device=self.device)
                    utils.reduce(t, reduction="mean")
                    reduced[k] = t.item()
                val_logs = reduced

            if self.is_main_process:
                val_logs_total.update({f"{part}_{k}": v for k, v in val_logs.items()})

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        if self.is_main_process:
            logs.update(val_logs_total)

        return logs

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
                disable=(not self.is_main_process)
            ):
                batch["epoch"] = epoch
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                )
            if self.is_main_process:
                self.writer.set_step(epoch * self.epoch_len, part)
                self._log_scalars(self.evaluation_metrics)
                self._log_batch(
                    batch_idx, batch, part
                )  # log only the last batch during inference

        return self.evaluation_metrics.result()

    def _monitor_performance(self, logs, not_improved_count):
        """
        Check if there is an improvement in the metrics. Used for early
        stopping and saving the best checkpoint.

        Args:
            logs (dict): logs after training and evaluating the model for
                an epoch.
            not_improved_count (int): the current number of epochs without
                improvement.
        Returns:
            best (bool): if True, the monitored metric has improved.
            stop_process (bool): if True, stop the process (early stopping).
                The metric did not improve for too much epochs.
            not_improved_count (int): updated number of epochs without
                improvement.
        """
        best = False
        stop_process = False
        if self.mnt_mode != "off":
            try:
                # check whether model performance improved or not,
                # according to specified metric(mnt_metric)
                if self.mnt_mode == "min":
                    improved = logs[self.mnt_metric] <= self.mnt_best
                elif self.mnt_mode == "max":
                    improved = logs[self.mnt_metric] >= self.mnt_best
                else:
                    improved = False
            except KeyError:
                self.logger.warning(
                    f"Warning: Metric '{self.mnt_metric}' is not found. "
                    "Model performance monitoring is disabled."
                )
                self.mnt_mode = "off"
                improved = False

            if improved:
                self.mnt_best = logs[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count >= self.early_stop:
                self.logger.info(
                    "Validation performance didn't improve for {} epochs. "
                    "Training stops.".format(self.early_stop)
                )
                stop_process = True
        return best, stop_process, not_improved_count

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader with some of the tensors on the device.
        """
        for tensor_for_device in self.cfg_trainer.device_tensors:
            if isinstance(batch[tensor_for_device], torch.Tensor):
                batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
            if isinstance(batch[tensor_for_device], list):
                for item in batch[tensor_for_device]:
                    if isinstance(item, torch.Tensor):
                        item = item.to(self.device)
                    if isinstance(item, dict):
                        for k, v in item.items():
                            item[k] = v.to(self.device)
        return batch

    def transform_batch(self, batch):
        """
        Transforms elements in batch. Like instance transform inside the
        BaseDataset class, but for the whole batch. Improves pipeline speed,
        especially if used with a GPU.

        Each tensor in a batch undergoes its own transform defined by the key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform).
        """
        # do batch transforms on device
        transform_type = "train" if self.is_train else "inference"
        transforms = self.batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                batch[transform_name] = transforms[transform_name](
                    batch[transform_name]
                )
        return batch

    def _clip_grad_norm(self):
        """
        Clips the gradient norm by the value defined in
        config.trainer.max_grad_norm
        """
        if self.config["trainer"].get("max_grad_norm", None) is not None:
            if self.accelerator is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(),
                                                 self.config["trainer"]["max_grad_norm"])
            else:
                clip_grad_norm_(
                    self.model.parameters(), self.config["trainer"]["max_grad_norm"]
                )

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        """
        Calculates the gradient norm for logging.

        Args:
            norm_type (float | str | None): the order of the norm.
        Returns:
            total_norm (float): the calculated norm.
        """
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _progress(self, batch_idx):
        """
        Calculates the percentage of processed batch within the epoch.

        Args:
            batch_idx (int): the current batch index.
        Returns:
            progress (str): contains current step and percentage
                within the epoch.
        """
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Abstract method. Should be defined in the nested Trainer Class.

        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        return NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker):
        """
        Wrapper around the writer 'add_scalar' to log all metrics.

        Args:
            metric_tracker (MetricTracker): calculated metrics.
        """
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _log_audio(self, batch_idx, batch, num_samples=None):
        if "audio" not in batch:
            return

        audio_list = batch["audio"]
        sources_list = [source["audio"] for source in batch["sources"]]
        sample_rate = getattr(self, "sample_rate", 16000)
        batch_size = len(audio_list)

        if num_samples is None or num_samples >= batch_size:
            indices = range(batch_size)
        else:
            indices = random.sample(range(batch_size), num_samples)

        for i in indices:
            name_prefix = f"train/audio_{batch_idx}_idx{i}"
            self.writer.add_audio(
                f"{name_prefix}_aug",
                audio_list[i].squeeze(0),
                sample_rate
            )

            if "audio_original" in batch:
                self.writer.add_audio(
                    f"{name_prefix}_orig",
                    batch["audio_original"][i].squeeze(0),
                    sample_rate
                )

            for source_idx, source in enumerate(sources_list):
                self.writer.add_audio(
                    f"{name_prefix}_source_{source_idx}",
                    source[i].squeeze(0),
                    sample_rate
                )

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """
        arch = type(self.model).__name__

        if self.accelerator is not None:
            model_to_save = self.accelerator.unwrap_model(self.model)
        else:
            model_to_save = self.model

        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": model_to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }

        if self.is_main_process:
            filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
            if not (only_best and save_best):
                torch.save(state, filename)
                # if self.config.writer.log_checkpoints:
                # self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
                self.logger.info(f"Saving checkpoint: {filename} ...")
            if save_best:
                best_path = str(self.checkpoint_dir / "model_best.pth")
                torch.save(state, best_path)
                if self.config.writer.log_checkpoints:
                    self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
                self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        Loads model, optimizer, scheduler states.
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)

        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration differs. "
                "This may yield errors when loading state_dict."
            )

        # unwrap model if using accelerator
        if self.accelerator is not None:
            model_to_load = self.accelerator.unwrap_model(self.model)
        else:
            model_to_load = self.model

        state_dict = align_state_dict_keys(
            checkpoint["state_dict"]
        )

        missing, unexpected = model_to_load.load_state_dict(state_dict, strict=False)
        if missing:
            self.logger.warning(f"Missing keys when loading state_dict: {missing}")
        if unexpected:
            self.logger.warning(f"Unexpected keys in state_dict: {unexpected}")

        # load optimizer/scheduler only if config matches
        if (
            checkpoint["config"]["optimizer"] != self.config["optimizer"]
            or checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"]
        ):
            self.logger.warning(
                "Optimizer or lr_scheduler config differs. "
                "Skipping their state restoration."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        self.logger.info(
            f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        )

    def _from_pretrained(self, pretrained_path):
        """
        Init model with weights from pretrained pth file.
        Only initializes the model (not optimizer/scheduler).
        """
        pretrained_path = str(pretrained_path)
        if hasattr(self, "logger"):
            self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            print(f"Loading model weights from: {pretrained_path} ...")

        checkpoint = torch.load(pretrained_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        if self.accelerator is not None:
            model_to_load = self.accelerator.unwrap_model(self.model)
        else:
            model_to_load = self.model

        state_dict = align_state_dict_keys(state_dict)
        missing, unexpected = model_to_load.load_state_dict(state_dict, strict=False)

        if missing:
            self.logger.warning(f"Missing keys when loading state_dict: {missing}")
        if unexpected:
            self.logger.warning(f"Unexpected keys in state_dict: {unexpected}")

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        self.logger.info(f"Model weights successfully loaded from {pretrained_path}")

    def _push_to_hugging_face(self):
        if not self.is_main_process:
            return

        writer_id = getattr(self.writer, "get_experiment_id", lambda: None)()

        hf_info = ModelLoader.push_to_hugging_face(
            repo_and_branch=self.config.trainer.get("hugging_face_repo"),
            checkpoint_dir=self.checkpoint_dir,
            writer_id=writer_id,
            logger=self.logger
        )
        if hf_info is not None:
            repo_id, branch, commit_id = hf_info

            if hasattr(self.writer, "add_huggingface_info"):
                self.writer.add_huggingface_info(repo_id, branch, commit_id)
