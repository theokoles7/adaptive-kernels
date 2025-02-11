"""Cifar 10 dataset and utilities."""

__all__ = ["Cifar10"]

from json                   import dumps
from logging                import Logger
from typing                 import override

from torch.utils.data       import DataLoader
from torchvision.datasets   import CIFAR10
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from datasets.__base__      import Dataset
from utils                  import LOGGER

class Cifar10(Dataset):
    """The CIFAR-10 dataset (https://www.cs.toronto.edu/~kriz/cifar.html) consists of 60000 32x32 
    colour images in 10 classes, with 6000 images per class. There are 50000 training images and 
    10000 test images.
    """

    @override
    def __init__(self, 
        batch_size: int,
        data_path:  str =   "data",
        **kwargs
    ):
        """# Initialize Cifar10 dataset loaders.

        ## Args:
            * batch_size    (int):              Dataset batch size.
            * data_path     (str, optional):    Path at which dataset is located/can be downloaded. 
                                                Defaults to "./data/".
        """
        # Initialize parent class
        super(Cifar10, self)
        
        # Initialize logger
        self.__logger__:        Logger =        LOGGER.getChild(suffix = 'cifar10-dataset')
        
        # Create transform
        transform:              Compose =       Compose([
                                                    # Resize images to 32x32 pixels
                                                    Resize(32),
                                                    
                                                    # Convert images to PyTorch tensors
                                                    ToTensor(),
                                                    
                                                    # Normalize pixel values
                                                    Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
                                                ])
        self.__logger__.debug(f"Initialized data transform:\n{dumps(vars(transform), indent = 2, default = str)}")

        # Download/verify train data
        train_data:             CIFAR10 =       CIFAR10(
                                                    root =      data_path,
                                                    download =  True,
                                                    train =     True,
                                                    transform = transform
                                                )
        self.__logger__.debug(f"Initialized training data:\n{dumps(vars(train_data), indent = 2, default = str)}")

        # Download/verify test data
        test_data:              CIFAR10 =       CIFAR10(
                                                    root =      data_path,
                                                    download =  True,
                                                    train =     False,
                                                    transform = transform
                                                )
        self.__logger__.debug(f"Initialized testing data:\n{dumps(vars(test_data), indent = 2, default = str)}")

        # Create training loader
        self._train_loader_:    DataLoader =    DataLoader(
                                                    dataset =       train_data,
                                                    batch_size =    batch_size,
                                                    pin_memory =    True,
                                                    num_workers =   4,
                                                    shuffle =       True,
                                                    drop_last =     True
                                                )
        self.__logger__.debug(f"Initialized training data loader:\n{dumps(vars(self._train_loader_), indent = 2, default = str)}")

        # Create testing loader
        self._test_loader_:     DataLoader =    DataLoader(
                                                    dataset =       test_data,
                                                    batch_size =    batch_size,
                                                    pin_memory =    True,
                                                    num_workers =   4,
                                                    shuffle =       True,
                                                    drop_last =     False
                                                )
        self.__logger__.debug(f"Initialized testing data loader:\n{dumps(vars(self._test_loader_), indent = 2, default = str)}")

        # Define parameters (passed to model during initialization for layer dimensions)
        self._classes_:     int =           10
        self._channels_:    int =           3
        self._dimension_:   int =           32
        self.__logger__.debug(f"Attributes defined: _num_classes_ = {self._classes_}, _channels_in_ = {self._channels_}, _dim_ = {self._dimension_}")
    
    def __str__(self) -> str:
        """# Provide string format of Cifar10 dataset object.

        ## Returns:
            * str:  String format of Cifar10 dataset
        """
        return f"Cifar10 dataset ({self._classes_} classes)"