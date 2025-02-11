# Running Experiments
[< Commands](./README.md)

Experiments can be run on any combination of:
* Datasets: `cifar10`, `cifar100`, `imagenet`, `mnist`
* Models:   `normal-cnn`, `resnet`, `vgg`
* Kernels:  `cauchy`, `gaussian`, `gumbel`, `laplace`
* Kernel Groups: 1-14

***NOTE***: An experiment is simply a series of jobs

In order to run an experiment, simply run the following command using at least one dataset and model, and one or more kernel and kernel group:

```bash
python -m main run-experiment --datasets cifar10 cifar100 --models normal-cnn resnet --kernels cauchy gaussian --kernel-groups 7 13
```

## Process

The experiment process will take into account the combination of options provided for datasets, models, kernels, and groups and run a unique experiment with each possible combination of those parameters. The experiment statistics will record statistics from each job, as well as keep record of the maximum performance on each dataset. At the conclusion of the experiment, a file will be saved containing the statistics:
```json
{
    "parameters":                   locals(),
    "jobs":                         {},
    "dataset_performance_records":  {
                                        dataset: {
                                            "model":        "",
                                            "kernel":       "",
                                            "kernel-group": 0,
                                            "accuracy":     0,
                                            "model":        ""
                                        } for dataset in datasets
                                    }
}
```

In addition to the experiment statistics file, a statistics file for each job will be saved in the same directory for analysis.

## Usage
```text
usage: adaptive-kernel run-experiment [-h] [--epochs EPOCHS] [--datasets {cifar10,cifar100,imagenet,mnist} [{cifar10,cifar100,imagenet,mnist} ...]] [--data-path DATA_PATH] [--batch-size BATCH_SIZE] [--models {normal-cnn,resnet,vgg} [{normal-cnn,resnet,vgg} ...]] [--learning-rate LEARNING_RATE]
                                      [--kernels {cauchy,gaussian,gumbel,laplace} [{cauchy,gaussian,gumbel,laplace} ...]] [--location LOCATION] [--scale SCALE] [--kernel-size KERNEL_SIZE] [--kernel-groups {1,2,3,4,5,6,7,8,9,10,11,12,13,14} [{1,2,3,4,5,6,7,8,9,10,11,12,13,14} ...]]

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of epochs for which training phase of job will execute. Defaults to 200.

Datasets:
  --datasets {cifar10,cifar100,imagenet,mnist} [{cifar10,cifar100,imagenet,mnist} ...]
                        Dataset(s) on which experiments will be executed.
  --data-path DATA_PATH
                        Path at which dataset will be downloaded/loaded. Defaults to "./data/".
  --batch-size BATCH_SIZE
                        Dataset batch size for training phase. Defaults to 64.

Models:
  --models {normal-cnn,resnet,vgg} [{normal-cnn,resnet,vgg} ...]
                        Model(s) on which experiments will be executed.
  --learning-rate LEARNING_RATE, -lr LEARNING_RATE
                        Model's learning rate. Defaults to 0.1.

Kernels:
  --kernels {cauchy,gaussian,gumbel,laplace} [{cauchy,gaussian,gumbel,laplace} ...]
                        Kernel(s) on which experiments will be executed. Equivalent of experiment(s) with no kernel will automatically be executed for comparison.
  --location LOCATION, -mu LOCATION
                        Location (Mu/Mean) parameter. Defaults to 0.
  --scale SCALE, -sigma SCALE
                        Scale (Sigma/Variance) parameter. Defaults to 1.
  --kernel-size KERNEL_SIZE
                        Kernel size (square). Defaults to 3.
  --kernel-groups {1,2,3,4,5,6,7,8,9,10,11,12,13,14} [{1,2,3,4,5,6,7,8,9,10,11,12,13,14} ...]
                        Kernel configuration type. Defaults to 13.
```