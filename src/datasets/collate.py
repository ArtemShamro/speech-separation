import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    specs = [item["spectrogram"].squeeze(0).transpose(0, 1)
             for item in dataset_items]  # list [1, 128, L]

    specs_lens = torch.tensor([item["spectrogram"].shape[-1]
                               for item in dataset_items])  # list [1, 1]

    phases = [item["phase"].squeeze(0).transpose(0, 1)
              for item in dataset_items]

    audio = [item["audio"].squeeze(0) for item in dataset_items]
    audio_lens = [item["audio"].shape[-1] for item in dataset_items]

    audio_original = [item["audio_original"] for item in dataset_items]

    # batching 321
    audio_batch = torch.nn.utils.rnn.pad_sequence(
        audio, batch_first=True, padding_value=0.0)
    specs_batch = torch.nn.utils.rnn.pad_sequence(
        specs, batch_first=True, padding_value=0.0).transpose(-2, -1)
    phases_batch = torch.nn.utils.rnn.pad_sequence(
        phases, batch_first=True, padding_value=0.0).transpose(-2, -1)

    audio_path = [item["audio_path"] for item in dataset_items]

    video_batch = torch.stack([
        torch.stack([
            torch.tensor(item["mouth1"]) if item["mouth1"] is not None else torch.zeros_like(torch.tensor(item["mouth2"])),
            torch.tensor(item["mouth2"]) if item["mouth2"] is not None else torch.zeros_like(torch.tensor(item["mouth1"]))
        ])
        for item in dataset_items
    ])

    sources_list = []
    n_sources = len(dataset_items[0]['sources'])

    for source_idx in range(n_sources):
        source_specs = [item["sources"][source_idx]["spectrogram"].squeeze(0).transpose(0, 1)
                        for item in dataset_items]
        source_audio = [item["sources"][source_idx]["audio"].squeeze(0) for item in dataset_items]
        source_phases = [item["sources"][source_idx]["phase"].squeeze(0).transpose(0, 1)
                         for item in dataset_items]

        source_audio_batch = torch.nn.utils.rnn.pad_sequence(
            source_audio, batch_first=True, padding_value=0.0)
        source_specs_batch = torch.nn.utils.rnn.pad_sequence(
            source_specs, batch_first=True, padding_value=0.0).transpose(-2, -1)
        source_phases_batch = torch.nn.utils.rnn.pad_sequence(
            source_phases, batch_first=True, padding_value=0.0).transpose(-2, -1)

        sources_list.append({
            "audio": source_audio_batch,
            "spectrogram": source_specs_batch,
            "phase": source_phases_batch,
        })

    return {
        "spectrogram": specs_batch,
        "phase": phases_batch,
        "spectrogram_length": specs_lens,
        "audio": audio_batch,
        "video": video_batch,
        "audio_lengths": audio_lens,
        "sources": sources_list,
        "audio_original": audio_original,
        "audio_path": audio_path,
        "audio_length": audio_batch.shape[-1],
    }
