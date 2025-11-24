import os
import zipfile
from pathlib import Path
from torch import Tensor
import torch_audiomentations as ta
import urllib.request
from src.utils.io_utils import ROOT_PATH
from .base_aug import BaseAugmentation


class ApplyImpulseResponse(BaseAugmentation):
    """
    Apply a random room impulse response (RIR) from the RIRS_NOISES dataset
    to simulate reverberation effects.
    """

    URL = "https://www.openslr.org/resources/28/rirs_noises.zip"

    def __init__(self, data_dir=None, temp_dir=None, p=0.3, *args, **kwargs):
        """
        Initialize the augmentation, downloading RIRS_NOISES if needed.

        Args:
            data_dir (str | Path, optional): Directory to store or search for RIRs.
            temp_dir (str | Path, optional): Temporary extraction directory.
            p (float): Probability of applying the augmentation.
        """
        super().__init__()

        if data_dir is None:
            data_dir = Path(ROOT_PATH / "data" / "datasets"
                            / "rirs_noises" / "simulated_rirs")
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = Path(data_dir)

        if temp_dir is None:
            temp_dir = Path(ROOT_PATH / "data" / "temp" / "rirs_noises")
            temp_dir.mkdir(exist_ok=True, parents=True)
        self._temp_dir = Path(temp_dir)

        rir_path = self._find_existing_rirs(self._data_dir)
        if rir_path is None:
            rir_path = self._find_existing_rirs(self._temp_dir)

        if rir_path is None:
            print("[ApplyImpulseResponse] No local RIRS found, downloading...")
            rir_path = self._download_and_extract()

        self._aug = ta.ApplyImpulseResponse(ir_paths=[rir_path], p=p, *args, **kwargs)

    def _find_existing_rirs(self, base_dir: Path):
        """
        Look for a directory containing .wav impulse responses.

        Args:
            base_dir (Path): Path to search.

        Returns:
            str | None: Directory path if found, else None.
        """
        if not base_dir.exists():
            return None
        for root, _, files in os.walk(base_dir):
            if any(f.endswith(".wav") for f in files):
                return root
        return None

    def _download_and_extract(self) -> str:
        """
        Download and extract the RIRS_NOISES dataset.

        Returns:
            str: Path to directory containing .wav impulse responses.
        """
        archive_path = self._temp_dir / "rirs_noises.zip"
        print(f"[ApplyImpulseResponse] Downloading from {self.URL} ...")
        urllib.request.urlretrieve(self.URL, str(archive_path))

        print("[ApplyImpulseResponse] Extracting archive...")
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(self._temp_dir)

        # ищем папку с wav файлами
        rir_root = self._find_existing_rirs(self._temp_dir)
        if rir_root is None:
            raise RuntimeError("RIRS_NOISES: no .wav files found after extraction!")

        print(f"[ApplyImpulseResponse] Found RIR WAVs in: {rir_root}")
        return rir_root

    def __call__(self, data: Tensor):
        """
        Apply a random impulse response to simulate reverberation.

        Args:
            data (Tensor): Input waveform tensor [B, T].

        Returns:
            Tensor: Augmented waveform tensor [B, T].
        """
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
