# Fasted Gradient Signed Method

FSGM (Fasted Gradient Signed Method) is white-box adversarial attack on deep neural network classifiers.

## Installation

```console
git clone https://github.com/bareform/dependencies.git
conda update -n base -c defaults conda
conda env create -f dependencies/environment.yml
conda activate bareform

git clone https://github.com/bareform/fgsm.git
cd fgsm
```

<details>
  <summary> Dependencies (click to expand) </summary>
  
  ## Dependencies
  - Python 3.10
  - datasets
  - numpy
  - torch
  - torchvision

</details>

## Quick Start

To generate 50 adversarial examples from ImageNet-1k bullfrog images to fool a pretrained ResNet-18 classifier:

```
python3 -m utils.trainer --config="./configs/imagenet-1k-bullfrog.toml"
```

Using the configuration provided at `./configs/imagenet-1k-bullfrog.toml`, we drop the classification accuracy to 0% from 86%.

## Method

Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy

Google

> Several machine learning models, including neural networks, consistently misclassify adversarial examples - inputs formed by applying small but intentionally worst-case perturbations to examples from the dataset, such that the perturbed input results in the model outputting an incorrect answer with high confidence. Early attempts at explaining this phenomenon focused on nonlinearity and overfitting. We argue instead that the primary cause of neural networks' vulnerability to adversarial perturbation is their linear nature. This explanation is supported by new quantitative results while giving the first explanation of the most intriguing fact about them: their generalization across architectures and training sets. Moreover, this view yields a simple and fast method of generating adversarial examples. Using this approach to provide examples for adversarial training, we reduce the test set error of a maxout network on the MNIST dataset.

## Citation

The original paper can be found at:
```
@misc{goodfellow2014explaining,
    title={Explaining and Harnessing Adversarial Examples},
    author={Ian J. Goodfellow and Jonathon Shlens and Christian Szegedy},
    year={2014},
    eprint={1412.6572},
    archivePrefix={arXiv},
    primaryClass={stat.ML}
}
```
