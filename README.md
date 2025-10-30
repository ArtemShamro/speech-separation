# Speech source separation with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository provides an infrastructure template for developing speech source separation models using PyTorch.

**Demo:**  
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

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

