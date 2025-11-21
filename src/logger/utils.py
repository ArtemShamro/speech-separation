import io

import matplotlib.pyplot as plt
import numpy as np
import PIL
from torchvision.transforms import ToTensor

plt.switch_backend("agg")  # fix RuntimeError: main thread is not in main loop


def plot_images(imgs, config):
    """
    Combine several images into one figure.

    Args:
        imgs (Tensor): array of images (B X C x H x W).
        config (DictConfig): hydra experiment config.
    Returns:
        image (Tensor): a single figure with imgs plotted side-to-side.
    """
    # name of each img in the array
    names = config.writer.names
    # figure size
    figsize = config.writer.figsize
    fig, axes = plt.subplots(1, len(names), figsize=figsize)
    for i in range(len(names)):
        # channels must be in the last dim
        img = imgs[i].permute(1, 2, 0)
        axes[i].imshow(img)
        axes[i].set_title(names[i])
        axes[i].axis("off")  # we do not need axis
    # To create a tensor from matplotlib,
    # we need a buffer to save the figure
    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    # convert buffer to Tensor
    image = ToTensor()(PIL.Image.open(buf))

    plt.close()

    return image


def plot_spectrogram(spectrogram, name=None):
    """
    Plot spectrogram with power-to-dB conversion (no librosa).
    Args:
        spectrogram (Tensor): power spectrogram tensor (|X|^2).
        name (None | str): optional title.
    Returns:
        image (Tensor): RGB image tensor.
    """
    spec = spectrogram.detach().cpu().numpy()

    ref = np.max(spec)
    eps = 1e-10
    spec_db = 10.0 * np.log10(np.maximum(spec, eps) / ref)

    spec_norm = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-9)

    plt.figure(figsize=(20, 5))
    plt.pcolormesh(spec_norm, cmap="magma")
    if name:
        plt.title(name)
    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    plt.close()

    image = ToTensor()(PIL.Image.open(buf).convert("RGB"))
    return image


def align_state_dict_keys(state_dict):
    """
    Removes prefixes from state dict
    """
    new_state = {}
    for k, v in state_dict.items():
        k = (
            k.replace("module.", "")
            .replace("model.", "")
            .replace("compile.", "")
            .replace("_orig_mod.", "")
        )
        new_state[k] = v
    return new_state
