import comet_ml
from pathlib import Path

import pandas as pd

import random

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        if self.accelerator is not None:
            with self.accelerator.autocast():
                outputs = self.model(**batch)
                batch.update(outputs)
                all_losses = self.criterion(**batch)
                batch.update(all_losses)
        else:
            outputs = self.model(**batch)
            batch.update(outputs)
            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        if self.is_train:
            if self.accelerator is not None:
                self.accelerator.backward(batch["loss"])
            else:
                batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example
        if not self.is_main_process:
            return
        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch, num_samples=self.train_log_items)
        else:
            # Log Stuff
            self.log_spectrogram(**batch, num_samples=self.train_log_items)
            self.log_predictions(**batch, examples_to_log=self.train_log_items)

    def log_spectrogram(self, spectrogram, num_samples=4, **batch):
        """
        Logs several spectrograms from the batch.

        Args:
            spectrogram (Tensor): [B, F, T] or [B, 1, F, T] tensor
            num_samples (int): how many examples from the batch to log
        """
        batch_size = spectrogram.size(0)

        if num_samples is None or num_samples >= batch_size:
            indices = range(batch_size)
        else:
            indices = random.sample(range(batch_size), num_samples)

        for i in indices:
            spec = spectrogram[i].detach().cpu()

            if spec.ndim == 3 and spec.size(0) == 1:
                spec = spec.squeeze(0)

            image = plot_spectrogram(spec)
            image_np = (image.permute(1, 2, 0).numpy() * 255).astype("uint8")

            name_prefix = f"spectrogram/batch_{i}"
            self.writer.add_image(name_prefix, image_np)

    def log_predictions(
        self, audio, sources, preds, batch_permuts, epoch, examples_to_log=10, **batch
    ):

        n_sources = len(preds)
        B = audio.shape[0]
        sample_rate = getattr(self, "sample_rate", 16000)

        for elem_idx in range(B):
            name_prefix = f"val/audio_idx:{elem_idx}_mix"
            mix_audio = audio[elem_idx]
            self.writer.add_audio(
                f"{name_prefix}",
                mix_audio.squeeze(0),
                metadata={
                    "context": f"val_{elem_idx}",
                },
                sample_rate=sample_rate,
            )
            for source_idx in range(n_sources):
                name_prefix = f"val/audio_idx:{elem_idx}_s:{source_idx}_pred"
                predicted_source = preds[source_idx]["audio"][elem_idx]
                self.writer.add_audio(
                    f"{name_prefix}",
                    predicted_source.squeeze(0),
                    metadata={
                        "context": f"val_{elem_idx}",
                    },
                    sample_rate=sample_rate,
                )

                name_prefix = f"val/audio_idx:{elem_idx}_s:{source_idx}_orig"
                original_source = sources[batch_permuts[elem_idx, source_idx]]["audio"][
                    elem_idx
                ]
                self.writer.add_audio(
                    f"{name_prefix}",
                    original_source.squeeze(0),
                    metadata={
                        "context": f"val_{elem_idx}",
                    },
                    sample_rate=sample_rate,
                )
