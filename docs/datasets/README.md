# Dataserts
[< Documentation](../README.md)

## Contents:
* [Cifar-10](#cifar-10)
* [Cifar-100](#cifar-100)
* [ImageNet](#imagenet-tiny200)
* [MNIST](#mnist)

## Cifar-10

The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

### Specifications
* Channels: 3
* Classes: 10
* Dimension: 32

## Cifar-100

This [dataset](https://www.cs.toronto.edu/~kriz/cifar.html) is just like the CIFAR-10, except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).

### Specifications
* Channels: 3
* Classes: 100
* Dimension: 32

## ImageNet (Tiny200)

[ImageNet](https://image-net.org/) is an image database organized according to the WordNet hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images. The project has been instrumental in advancing computer vision and deep learning research. The data is available for free to researchers for non-commercial use.

### Specifications
* Channels: 3
* Classes: 200
* Dimension: 32

## MNIST

The [MNIST](https://paperswithcode.com/dataset/mnist) database (Modified National Institute of Standards and Technology database) is a large collection of handwritten digits. It has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger NIST Special Database 3 (digits written by employees of the United States Census Bureau) and Special Database 1 (digits written by high school students) which contain monochrome images of handwritten digits. The digits have been size-normalized and centered in a fixed-size image. The original black and white (bilevel) images from NIST were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

### Specifications
* Channels: 1
* Classes: 10
* Dimension: 16