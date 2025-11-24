from pathlib import Path
import torchaudio
from tqdm import tqdm
import json

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, data_path, mouth_path=None, temp_dir=None, reindex=False, dataset_name: str = "custom_dataset", part: str = "train", *args, **kwargs):
        self.dataset_name = dataset_name
        data_path = Path(data_path)

        if temp_dir is None:
            self._temp_dir = Path(ROOT_PATH / "data" / "datasets" / dataset_name)
            self._temp_dir.mkdir(parents=True, exist_ok=True)

        if not data_path.exists():
            raise ValueError(f"Audio path does not exist: {data_path}")
        if not data_path.is_dir():
            raise ValueError(f"Audio path is not a directory: {data_path}")
        
        self._data_dir = data_path
        self._mix_dir = self._data_dir / "mix"
        self._sources = [p for p in self._data_dir.iterdir() if p.name != "mix"]
        self._mouth_path = Path(mouth_path) if mouth_path is not None else None

        if self._mouth_path is not None:
            if not self._mouth_path.exists():
                raise ValueError(f"Mouth path does not exist: {data_path}")
            if not self._mouth_path.is_dir():
                raise ValueError(f"Mouth path is not a directory: {data_path}")

        assert self._mix_dir.exists(), f"Mix directory with audio not found: {self._mix_dir}"

        index = self._get_or_load_index(reindex)

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, reindex=False):
        index_path = self._temp_dir / f"{self.dataset_name}_index.json"
        if index_path.exists() and not reindex:
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index()
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self):
        index = []
        audio_files = sorted(
            [p for p in self._mix_dir.iterdir() if p.suffix.lower() in [".wav", ".flac", ".mp3", ".m4a"]]
        )

        for audio_file in tqdm(audio_files, desc="Preparing custom dataset"):
            try:
                info = torchaudio.info(str(audio_file))
                audio_len = info.num_frames / info.sample_rate
            except Exception as e:
                print(f"Warning: failed to load {audio_file}: {e}")
                continue

            if self._mouth_path is not None:
                mouth1, mouth2 = str(audio_file.stem).split("_")
                mouth1_path = self._mouth_path / Path(f"{mouth1}.npz")
                mouth2_path =  self._mouth_path / Path(f"{mouth2}.npz")
                mouth1_path = str(mouth1_path.absolute().resolve())
                mouth2_path = str(mouth2_path.absolute().resolve())
            else:
                mouth1_path = mouth2_path = None

            index_element = {
                "mix_path": str(audio_file.absolute().resolve()),
                "audio_len": audio_len,
                "mouth1_path": mouth1_path,
                "mouth2_path": mouth2_path
            }

            for source_dir in self._sources:
                index_element.update({f"{source_dir.name}_path": str(
                    (source_dir / audio_file.name).absolute().resolve())})

            index.append(index_element)

        return index