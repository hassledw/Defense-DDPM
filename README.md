# GenAI-ImageDenoise

## Introduction
Adversarial attacks like FGSM, PGD, or JSMA on deep learning image classifiers pose a threat to the reliability of image classification models by subtly altering the source image, causing intentional misclassifications. These modifications, which can be imperceptible, have the potential to compromise the security of sensitive or classified information, resulting in adverse consequences. For example, running an FGSM attack on a face recognition application can change the label of the face, with minimal perturbations, causing an adversary to access sensitive information of someone else.

Currently, there are not any SoTA defense methods for this domain, as removing attacked image noise is a very challenging task. In this area of research, previous generative AI model implementations were only adequately successful at adversarial image defense, or have harsh limitations.

Our goal is to create a robust defense against these challenging attacks on all image classification domains by manipulating and using a more modern generative AI model, Stable Diffusion, to pass all “attacked” images and an optimal guided prompt as input to generate “cleansed” images.

## Datasets
* [Birds 525 Species- Image Classification](https://www.kaggle.com/datasets/gpiosenka/100-bird-species?resource=download): A collection of ~90,000 labeled bird images. 525 species, 84635 train, 2625 test, 2625 validation images of dimension 224X224X3 jpg.

## Repo Contents
* **birdclassifier.py**: is the [Rexnet](https://huggingface.co/docs/timm/en/models/rexnet) classifier fine-tuning model code for the birds dataset. 