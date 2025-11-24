# Audio-Visual Speech Separation with DTTNet

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#links">Links</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains an **audio-visual speech separation system** based on a lightweight but high-performance architecture **DTTNet (Chen et al., ICASSP 2024)**, extended with multiple fusion mechanisms for integrating lip-motion video features into the speech-separation pipeline.

The project investigates how visual information improves source separation in single-channel speech mixtures and shows that multimodal conditioning significantly boosts SI-SNRi while keeping the model compact.

### **Model Architecture**

**Base model: DTTNet**
- Encoder–Latent–Decoder UNet-like structure
- TFC–TDF blocks for spectral modeling
- Improved Dual-Path (IDP) modules
- Multi-head channel splitting
- Reconstruction using magnitude/phase masks and inverse STFT

**Audio-Visual Extensions**
- **FiLM Fusion:** Feature-wise Linear Modulation for audio–video conditioning (channel- and frequency-wise)
- **Cross-Attention Fusion:** Transformer-style interaction between audio and video streams
- **VPBlock + CAF:** Video-Prior and Channel-Attention Fusion modules adopted from RTFS-Net

### **Training**
- **Permutation-Invariant Training (PIT)**
- STFT: `n_fft=1024`, `hop_length=256`
- AdamW (`lr=1e-3`, weight decay `1e-5`, cosine LR + warmup)
- Augmentations: reverb, gain, speed, MUSAN noise, SpecAugment
- Hardware: A100, bf16
- Batch sizes: 64 (FiLM), 16 (Cross-Attention)

### **Results**

| Model | SI-SNRi | PESQ | STOI |
|-------|---------|-------|-------|
| DTT-Net Audio 15M | 6.17 | – | – |
| DTT-Net Audio+Video (FiLM) 25M | 7.20 | – | – |
| DTT-Net Audio+Video (Cross Attention) 60M | 7.26 | 1.66 | 0.82 |
| DTT-Net Audio+Video (Cross Attention) 200M | **8.64** | **1.87** | **0.85** |

Audio-visual fusion improves SI-SNRi by **up to +2 dB**, accelerates convergence, and increases robustness.

### **Demo:**
A demo notebook is available at `src/notebooks/demo.ipynb`.

## Installation

Follow these steps to set up the project:

1. Clone the repository.

2. Create a file named .env in the project root using env_template as a reference.

3. Install dependencies.

- On Google Colab or Kaggle:

   ```bash
   cd {repo_folder}
   make
   ```

- Locally:

  ```bash
   cd {repo_folder}
   pip install uv
   uv venv
   source .venv/bin/activate
   uv sync
   ```

## How To Use

- To train a model, run the following command:

```bash
accelerate launch --dynamo_backend no LAUNCH_ARGUMENTS train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

- To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

## Links

[Video encoder source](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks)

[Pretrained checkpoints](https://huggingface.co/areshx/source-sep)

## Credits
This project builds on research and ideas from:
* DTTNet (Chen et al., ICASSP 2024)
* FiLM (Perez et al., 2018)
* Transformers (Vaswani et al., 2017)
* RTFS-Net fusion modules (Xiaoxue Li et al., 2022)

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

