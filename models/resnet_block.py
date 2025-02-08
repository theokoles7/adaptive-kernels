"""Resnet model block."""

__all__ = ["ResnetBlock"]

from json                   import dumps
from logging                import Logger

from torch                  import mean, no_grad, std, Tensor
from torch.nn               import BatchNorm2d, Conv2d, Module, Sequential
from torch.nn.functional    import relu

from kernels                import load_kernel
from utils                  import ARGS, LOGGER

class ResnetBlock(Module):
    """Block component for Resnet Model."""

    def __init__(self, 
        channels_in:    int, 
        channels_out:   int, 
        stride:         int,
        kernel:         str =   None,
        kernel_group:   int =   13,
        location:       float = 0.0,
        scale:          float = 1.0,
    ):
        """# Initialize Resnet block.

        ## Args:
            * channels_in   (int):              Input channels.
            * channels_out  (int):              Output channels.
            * stride        (int):              Convolution stride.
            * kernel        (str, optional):    Kernel with which model will be set.
            * kernel_group  (int, optional):    Kernel configuration type. Defaults to 13.
            * location      (float, optional):  Distribution location parameter. Defaults to 0.0.
            * scale         (float, optional):  Distribution scale parameter. Defaults to 1.0.
        """
        super(ResnetBlock, self).__init__()

        # Initialize logger
        self.__logger__:            Logger =        LOGGER.getChild(suffix = 'resnet-block')
        
        # Log initialization for debugging
        self.__logger__.debug(f"Initializing...\nParameters: {dumps(obj = locals(), indent = 2, default = str)}")
        
        # Initialize block layer attributes
        self._channels_out_:        int =           channels_out

        # Initialize distribution parameters
        self._kernel_:              str =           kernel
        self._kernel_group_:        int =           kernel_group
        self._locations_:           list[float] =   [location]*5
        self._scales_:              list[float] =   [scale]*5

        # Batch normalization layers
        self._bn1_:                 BatchNorm2d =   BatchNorm2d(
                                                        num_features =      channels_out
                                                    )
        self._bn2_:                 BatchNorm2d =   BatchNorm2d(
                                                        num_features =      channels_out
                                                    )

        # Convolving layers
        self._conv1_:               Conv2d =        Conv2d(
                                                        in_channels =       channels_in,
                                                        out_channels =      channels_out,
                                                        kernel_size =       3,
                                                        stride =            stride,
                                                        padding =           1,
                                                        bias =              False
                                                    )
        self._conv2_:               Conv2d =        Conv2d(
                                                        in_channels =       channels_out,
                                                        out_channels =      channels_out,
                                                        kernel_size =       3,
                                                        stride =            1,
                                                        padding =           1,
                                                        bias =              False
                                                    )

        # Shortcut layer
        self._shortcut_:            Sequential =    Sequential()

        # If stride is not 1 or input channels is not the same as output channels...
        if stride != 1 or channels_in != channels_out:
            
            # Set shortcut to True
            self._shortcut_kernel_: bool =          True
            
            # Define shortcut
            self._shortcut_:        Sequential =    Sequential(
                                                        Conv2d(
                                                            in_channels =   channels_in,
                                                            out_channels =  channels_out,
                                                            kernel_size =   1,
                                                            stride =        stride,
                                                            bias =          False
                                                        ),
                                                        BatchNorm2d(
                                                            num_features =  channels_out
                                                        )
                                                    )

    def forward(self, 
        X:  Tensor
    ) -> Tensor:
        """# Feed input through network and provide output.

        ## Args:
            * X (Tensor):   Input tensor

        ## Returns:
            * Tensor:   Output tensor
        """
        # INPUT LAYER =========================================================================
        # Log input shape for debugging
        self.__logger__.debug(f"Input shape: {X.shape}")

        # LAYER 1 =============================================================================
        # Pass through first convolving layer
        x1: Tensor =        self._conv1_(X)

        # Log output shape of first layer for debugging
        self.__logger__.debug(f"Layer 1 shape: {x1.shape}")

        # Without taking gradients...
        with no_grad(): 
            
            # Convert tensor to float values
            y:  Tensor =    x1.float()
            
            # Calculate mean & standard deviation of layer output
            self._locations_[0], self._scales_[0] = mean(y).item(), std(y).item()

        # LAYER 2 =============================================================================
        # Pass through second convolving layer
        x2: Tensor =        self._conv2_(relu(self._bn1_(self._kernel1_(x1) if self._kernel_ is not None else x1)))

        # Log output shape of first second for debugging
        self.__logger__.debug(f"Layer 2 shape: {x2.shape}")

        # Without taking gradients...
        with no_grad(): 
            
            # Convert tensor to float values
            y:  Tensor =    x2.float()
            
            # Calculate mean & standard deviation of layer output
            self._locations_[1], self._scales_[1] = mean(y).item(), std(y).item()

        # OUTPUT LAYER ========================================================================
        # Pass through batch normalization layer
        output: Tensor =    self._bn2_(self._kernel2_(x2) if self._kernel_ is not None else x2)

        # Log output shape of output layer for debugging
        self.__logger__.debug(f"Output layer shape: {output.shape}")

        # SHORTCUT LAYER ======================================================================
        # output += self._shortcut_(output)

        # Return output
        return relu(output)
    
    def set_kernels(self) -> None:
        """# Create/update kernels."""
        # Log for debugging
        self.__logger__.debug(f"Locations: {self._locations_}, Scales: {self._scales_}")

        # Set kernels
        for kernel, channel_size, location, scale in zip(
            ["_kernel1_", "_kernel2_"],
            [self._channels_out_]*2,
            self._locations_,
            self._scales_
        ):
            self.__setattr__(name = kernel, value = load_kernel(
                kernel =        self._kernel_,
                kernel_group =  self._kernel_group_,
                size =          3,
                channels =      channel_size,
                location =      location,
                scale =         scale
            ))