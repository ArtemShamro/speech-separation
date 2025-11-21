from pathlib import Path
from src.utils.io_utils import ROOT_PATH
from src.datasets.custom_dir_audio_dataset import CustomDirAudioDataset


class CustomDirDataset(CustomDirAudioDataset):
    def __init__(self, data_path=None, part=None, reindex=False, dataset_name: str = "custom_dataset", *args, **kwargs):
        if data_path is None:
            data_path = ROOT_PATH / "data" / "datasets" / "dla_dataset"
        data_path = Path(data_path)

        audio_dir = data_path / "audio"
        if part is not None:
            audio_dir = audio_dir / part

        mouth_dir = data_path / "mouths"

        super().__init__(
            data_path=audio_dir,
            mouth_dir=mouth_dir,
            reindex=reindex,
            dataset_name=dataset_name,
            *args,
            **kwargs
        )
