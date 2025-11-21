import os
import zipfile
from pathlib import Path
from torch import nn, Tensor
import torch_audiomentations as ta
import urllib.request
from src.utils.io_utils import ROOT_PATH
from .base_aug import BaseAugmentation


class AddBackgroundNoise(BaseAugmentation):
    """
    Add background noise from the MUSAN dataset.
    Downloads MUSAN if not available locally and mixes background noise
    (noise, music, or babble) into the input signal at a random SNR level.
    """

    URL = "https://openslr.trmal.net/resources/17/musan.tar.gz"

    def __init__(
        self,
        data_dir=None,
        temp_dir=None,
        snr_db=(15, 30),
        include_babble=True,
        p=0.5,
        *args,
        **kwargs,
    ):
        """
        Initialize the augmentation.

        Args:
            data_dir (str | Path, optional): Directory to store or find the MUSAN dataset.
            temp_dir (str | Path, optional): Temporary directory for extraction.
            snr_db (tuple): Range of SNR levels (min, max) in decibels.
            include_babble (bool): Whether to include speech babble noises.
            p (float): Probability of applying the augmentation.
        """
        super().__init__()

        if data_dir is None:
            data_dir = Path(ROOT_PATH / "data" / "datasets" / "musan")
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = Path(data_dir)

        if temp_dir is None:
            temp_dir = Path(ROOT_PATH / "data" / "temp" / "musan")
            temp_dir.mkdir(exist_ok=True, parents=True)
        self._temp_dir = Path(temp_dir)

        musan_root = self._find_existing_musan(self._data_dir)
        if musan_root is None:
            musan_root = self._find_existing_musan(self._temp_dir)

        if musan_root is None:
            print("[AddBackgroundNoise] No local MUSAN found, downloading...")
            musan_root = self._download_and_extract()

        noise_paths = [os.path.join(musan_root, "noise"), os.path.join(musan_root, "music")]
        if include_babble:
            noise_paths.append(os.path.join(musan_root, "speech"))
        noise_paths = [p for p in noise_paths if os.path.exists(p)]

        self._aug = ta.AddBackgroundNoise(
            background_paths=noise_paths,
            min_snr_in_db=snr_db[0],
            max_snr_in_db=snr_db[1],
            p=p,
            *args,
            **kwargs,
        )

    def _find_existing_musan(self, base_dir: Path):
        """
        Check if a valid MUSAN dataset exists in the given directory.

        Args:
            base_dir (Path): Directory to check.

        Returns:
            str | None: Path to MUSAN root if found, otherwise None.
        """
        if not base_dir.exists():
            return None
        for sub in ["noise", "music", "speech"]:
            sub_path = base_dir / sub
            if sub_path.exists() and any(str(f).endswith(".wav") for f in sub_path.rglob("*.wav")):
                return str(base_dir)
        return None

    def _download_and_extract(self) -> str:
        """
        Download and extract the MUSAN dataset.

        Returns:
            str: Path to the extracted dataset root directory.
        """
        archive_path = self._temp_dir / "musan.zip"
        print(f"[AddBackgroundNoise] Downloading from {self.URL} ...")
        urllib.request.urlretrieve(self.URL, str(archive_path))

        print("[AddBackgroundNoise] Extracting archive...")
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(self._temp_dir)

        # В архиве MUSAN обычно создается подкаталог musan/
        for candidate in self._temp_dir.iterdir():
            if candidate.is_dir() and (candidate / "noise").exists():
                print(f"[AddBackgroundNoise] Found MUSAN in: {candidate}")
                return str(candidate)

        raise RuntimeError("MUSAN: no valid structure (noise/music/speech) found after extraction!")

    def __call__(self, data: Tensor):
        """
        Apply background noise augmentation to the input tensor.

        Args:
            data (Tensor): Input waveform tensor of shape [B, T].

        Returns:
            Tensor: Augmented waveform tensor of shape [B, T].
        """
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
