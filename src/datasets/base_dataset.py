import logging
import random

import torch

import numpy as np
import torchaudio
from torch.utils.data import Dataset

from typing import List
from pathlib import Path
logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """
    _sources: List[Path] = []

    def __init__(
        self,
        index,
        target_sr=16000,
        limit=None,
        min_audio_length=None,
        max_audio_length=None,
        shuffle_index=False,
        instance_transforms=None,
        **kwargs,
    ):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            text_encoder (CTCTextEncoder): text encoder.
            target_sr (int): supported sample rate.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            max_audio_length (int): maximum allowed audio length.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        self._assert_index_is_valid(index)

        index = self._filter_records_from_dataset(
            index, min_audio_length, max_audio_length
        )
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        if not shuffle_index:
            index = self._sort_index(index)

        self._index: list[dict] = index

        self.target_sr = target_sr
        self.instance_transforms = instance_transforms
        self.scheduled_transforms = False

        if instance_transforms is not None:
            for transform in instance_transforms:
                if hasattr(transform, "epoch_set"):
                    self.scheduled_transforms = True

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]
        mix_path = data_dict["mix_path"]
        mix_audio = self.load_audio(mix_path)
        mix_audio_original = mix_audio.clone()

        # apply waw augs befor getting spec
        mix_audio = self.preprocess_audio(mix_audio, freeze_parameters=True)

        mix_magnitude, mix_phase = self.get_spectrogram(mix_audio)

        sources = []
        for source in self._sources:
            sorce_name = source.name

            source_path = data_dict[f"{sorce_name}_path"]
            source_audio = self.load_audio(source_path)
            source_audio = self.preprocess_audio(source_audio, consistent_only=True)
            source_magnitude, source_phase = self.get_spectrogram(source_audio)

            source_video_path = data_dict[f"{sorce_name}_video_path"]
            source_video = np.load(source_video_path)["data"]

            source_data = {
                "audio": source_audio,
                "spectrogram": source_magnitude,
                "phase": source_phase,
                "video": source_video
            }

            sources.append(source_data)

        instance_data = {
            "audio": mix_audio,
            "audio_original": mix_audio_original,
            "spectrogram": mix_magnitude,
            "phase": mix_phase,
            "sources": sources,
            "audio_path": mix_path,
        }

        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.target_sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def get_spectrogram(self, audio):
        """
        Special instance transform with a special key to
        get spectrogram from audio.

        Args:
            audio (Tensor): original audio.
        Returns:
            spectrogram (Tensor): spectrogram for the audio.
        """
        spec = self.instance_transforms["get_spectrogram"](audio)
        magnitude = spec.abs()
        phase = spec.angle()
        return magnitude, phase

    def preprocess_audio(self, audio, consistent_only=False, freeze_parameters=False):
        """
        Preprocess audio with instance transforms[audio].
        """
        if self.instance_transforms is not None:
            # consistent transforms (same for mix and sources)
            if "audio_consistent" in self.instance_transforms.keys():
                audio_tf = self.instance_transforms["audio_consistent"]
                audio = audio_tf.apply_and_freeze(audio, freeze_parameters)
            # mix-only transforms
            if "audio" in self.instance_transforms.keys() and not consistent_only:
                audio_tf = self.instance_transforms["audio"]
                audio = audio_tf(audio)
        return audio

    def preprocess_data(self, instance_data):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                if transform_name == "get_spectrogram" or transform_name == "audio" or transform_name == "audio_consistent":
                    continue  # skip special key
                instance_data[transform_name] = self.instance_transforms[
                    transform_name
                ](instance_data[transform_name])
        return instance_data

    @staticmethod
    def _filter_records_from_dataset(
        index: list,
        min_audio_length,
        max_audio_length,
    ) -> list:
        """
        Filter some of the elements from the dataset depending on
        the desired max_test_length or max_audio_length.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            min_audio_length (float): minimum allowed audio length (sec).
            max_audio_length (float): maximum allowed audio length (sec).
            min_text_length (int): minimum allowed text length (chars).
            max_text_length (int): maximum allowed text length (chars).
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset that satisfied the condition. The dict has
                required metadata information, such as label and object path.
        """
        initial_size = len(index)

        if max_audio_length is not None or min_audio_length is not None:
            audio_lengths = np.array([el["audio_len"] for el in index])

        if max_audio_length is not None:
            too_long_audio = audio_lengths >= max_audio_length
            _total = too_long_audio.sum()
            if _total > 0:
                logger.info(
                    f"{_total} ({_total / initial_size:.1%}) records are longer than "
                    f"{max_audio_length} sec. Excluding them."
                )
        else:
            too_long_audio = False

        if min_audio_length is not None:
            too_short_audio = audio_lengths < min_audio_length
            _total = too_short_audio.sum()
            if _total > 0:
                logger.info(
                    f"{_total} ({_total / initial_size:.1%}) records are shorter than "
                    f"{min_audio_length} sec. Excluding them."
                )
        else:
            too_short_audio = False

        records_to_filter = too_long_audio | too_short_audio

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total} ({_total / initial_size:.1%}) records from dataset"
            )

        return index

    @staticmethod
    def _assert_index_is_valid(index):
        """
        Check the structure of the index and ensure it satisfies the desired
        conditions.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        for entry in index:
            if "mix_path" not in entry:
                print(entry)
            assert "mix_path" in entry, (
                "Each dataset item should include field 'mix_path'" " - path to audio file."
            )
            assert "audio_len" in entry, (
                "Each dataset item should include field 'audio_len'"
                " - length of the audio."
            )

    @staticmethod
    def _sort_index(index):
        """
        Sort index by audio length.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        Returns:
            index (list[dict]): sorted list, containing dict for each element
                of the dataset. The dict has required metadata information,
                such as label and object path.
        """
        return sorted(index, key=lambda x: x["audio_len"])

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        """
        Shuffle elements in index and limit the total number of elements.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
        """
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
