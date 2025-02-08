# Running Jobs
[< Commands](./README.md)

Jobs are run a little different from esperiemnts in that each of the `dataset`, `model`, and `kernel` "parameters" are used as commands themselves. However, the options for each are uniform, and the `kernel` parameter is optional.

## Process
The job process will simply run on its own, where on completion, it will save a file to record the statistics:

```json
{
    "parameters":       locals(),
    "epochs":           {},
    "best_accuracy":    0,
    "best_epoch":       0,
    "test_accuracy":    0,
    "test_loss":        0
}
```

## Usage

```text
usage: adaptive-kernel run-job [-h] [--epochs EPOCHS] {cifar10,cifar100,imagenet,mnist} ...

positional arguments:
  {cifar10,cifar100,imagenet,mnist}
                        Dataset selection.
    cifar10             Use Cifar-10 dataset for job process.
    cifar100            Use Cifar-100 dataset for job process.
    imagenet            Use ImageNet dataset for job process.
    mnist               Use MNIST dataset for job process.

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of epochs for which training phase of job will execute. Defaults to 200.
```

### Dataset Usage:

```text
usage: adaptive-kernel run-job cifar10 [-h] [--data-path DATA_PATH] [--batch-size BATCH_SIZE] {normal-cnn,resnet,vgg} ...

positional arguments:
  {normal-cnn,resnet,vgg}
                        Model selection.
    normal-cnn          Use Normal CNN model for job process.
    resnet              Use ResNet model for job process.
    vgg                 Use VGG model for job process.

options:
  -h, --help            show this help message and exit
  --data-path DATA_PATH
                        Path at which dataset will be downloaded/loaded. Defaults to "./data/".
  --batch-size BATCH_SIZE
                        Dataset batch size for training phase. Defaults to 64.
```

### Model Usage:

```text
usage: adaptive-kernel run-job cifar10 normal-cnn [-h] [--learning-rate LEARNING_RATE] {cauchy,gaussian,gumbel,laplace} ...

positional arguments:
  {cauchy,gaussian,gumbel,laplace}
                        Kernel selection.
    cauchy              Use Cauchy kernel for job process.
    gaussian            Use Gaussian kernel for job process.
    gumbel              Use Gumbel kernel for job process.
    laplace             Use Laplace kernel for job process.

options:
  -h, --help            show this help message and exit
  --learning-rate LEARNING_RATE, -lr LEARNING_RATE
                        Model's learning rate. Defaults to 0.1.
```

### Kernel Usage:

```text
usage: adaptive-kernel run-job cifar10 normal-cnn gaussian [-h] [--location LOCATION] [--scale SCALE] [--kernel-size KERNEL_SIZE] [--kernel-group {1,2,3,4,5,6,7,8,9,10,11,12,13}]

options:
  -h, --help            show this help message and exit
  --location LOCATION, -mu LOCATION
                        Location (Mu/Mean) parameter. Defaults to 0.
  --scale SCALE, -sigma SCALE
                        Scale (Sigma/Variance) parameter. Defaults to 1.
  --kernel-size KERNEL_SIZE
                        Kernel size (square). Defaults to 3.
  --kernel-group {1,2,3,4,5,6,7,8,9,10,11,12,13}
                        Kernel configuration type. Defaults to 13.
```