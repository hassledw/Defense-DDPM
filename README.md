# GenAI-ImageDenoise
### Developers
* Daniel Hassler (hassledw@vt.edu)
* Matthew Dim (matthewdim@vt.edu)
* Austin Burcham (austindb@vt.edu)

## Introduction
Deep learning adversarial attacks such as Fast Gradient Sign Method (FGSM) or Projected Gradient Descent (PGD) are popular white-box attack frameworks that can severely damage the trustworthiness of any machine learning classifier. In this method, we propose Defense-DDPM, a black-box defense strategy that leverages the capabilities of Denoising Diffusion Probabilistic Models (DDPM) to iteratively refine denoised images. This strategy utilizes NL-Means and Wavelet Transform as preprocessing techniques before being fed into a DDPM, enhancing the denoising process.

## Datasets
* [Birds 525 Species- Image Classification](https://www.kaggle.com/datasets/gpiosenka/100-bird-species?resource=download): A collection of ~90,000 labeled bird images. 525 species, 84635 train, 2625 test, 2625 validation images of dimension 224X224X3 jpg.

## Repo Contents
### DDPM-Defense
* **ddpm-defense.py**: the main python script used to generate defended images. Includes code to run Wavelet, or NL-Means as preprocessing steps.
* **ddpm-analysis.ipynb**: the main jupyter notebook used for evaluating our defense/attacks.
* **ddpm-defense.ipynb**: supplementary material for the ddpm starter code found on PyTorch.

### NL-Means
* **./nlmeans**: source code for the NL-Means algorithm with some generated datasets.

### Wavelet Transform
* **./wavelet**: source code for the Wavelet Transform algorithm.

### Neccessities
* **analysis.ipynb**: our data, attack, and defense visualization notebook
* **attacks.py**: our PyTorch attack framework that automates the process of attacking a dataset and saving to respective folders.
* **birdclassifier.py**: is the [Rexnet](https://huggingface.co/docs/timm/en/models/rexnet) classifier fine-tuning model code for the birds dataset.
* **saved_models**: a folder containing the weights for fine-tuned classifiers on our datasets.
* **generate_csv.py**: a python script that automatically creates evaluation CSV files. This is an automated process of the experimental setup in our paper.
* **evaluation_csvs**: a folder containing all evaluation CSV files including results.

