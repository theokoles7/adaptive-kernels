"""Dataset arguments package."""

__all__ = ["cifar10", "cifar100", "imagenet", "mnist"]

from utils.arguments.datasets.cifar10   import add_cifar10_parser
from utils.arguments.datasets.cifar100  import add_cifar100_parser
from utils.arguments.datasets.imagenet  import add_imagenet_parser
from utils.arguments.datasets.mnist     import add_mnist_parser