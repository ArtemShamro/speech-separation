from itertools import repeat

from hydra.utils import instantiate

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed

import os

from omegaconf import OmegaConf
from copy import deepcopy
import accelerate


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_dataloaders(config, device, accelerator: accelerate.Accelerator | None = None):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        device (str): device to use for batch transforms.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    # dataloaders init
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        # dataset partition init
        # instance transforms are defined inside
        dataset = instantiate(config.datasets[dataset_partition])

        assert config.dataloader.batch_size <= len(dataset), (
            f"The batch size ({config.dataloader.batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )
        cpu_count = os.cpu_count()
        num_workers = 2
        if cpu_count is not None:
            num_workers = max(1, cpu_count // 2)

        if dataset_partition == "train":
            partition_dataloader = instantiate(
                config.dataloader.loader,
                num_workers=num_workers,
                dataset=dataset,
                drop_last=True,
                shuffle=True,
                collate_fn=collate_fn,
                worker_init_fn=set_worker_seed,
                batch_size=config.dataloader.batch_size,
            )
        else:
            partition_dataloader = instantiate(
                config.dataloader.loader,
                num_workers=num_workers,
                dataset=dataset,
                collate_fn=collate_fn,
                drop_last=False,
                shuffle=False,
                worker_init_fn=set_worker_seed,
                batch_size=config.dataloader.batch_size,
            )

        if accelerator is not None:
            partition_dataloader = accelerator.prepare(partition_dataloader)

        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms
