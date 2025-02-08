"""Basic CNN model."""

__all__ = ["NormalCNN"]

from json                   import dump, dumps
from logging                import Logger

from torch                  import mean, no_grad, std, Tensor
from torch.nn               import Conv2d, Linear, MaxPool2d, Module
from torch.nn.functional    import relu

from kernels                import load_kernel
from utils                  import LOGGER

class NormalCNN(Module):
    """Basic CNN model."""

    def __init__(self,
        channels_in:    int, 
        channels_out:   int, 
        dim:            int,
        kernel:         str =   None,
        kernel_group:   int =   13,
        location:       float = 0.0,
        scale:          float = 1.0,
        **kwargs
    ):
        """# Initialize Normal CNN model.

        ## Args:
            * channels_in   (int):              Input channels.
            * channels_out  (int):              Output channels.
            * dim           (int):              Dimension of image (relevant for reshaping, post-convolution).
            * kernel        (str, optional):    Kernel with which model will be set.
            * kernel_group  (int, optional):    Kernel configuration type. Defaults to 13.
            * location      (float, optional):  Distribution location parameter. Defaults to 0.0.
            * scale         (float, optional):  Distribution scale parameter. Defaults to 1.0.
        """
        # Initialize parent class
        super(NormalCNN, self).__init__()
        
        # Initialize logger
        self.__logger__:        Logger =        LOGGER.getChild(suffix = 'normal-cnn')
        
        # Log initialization parameters for debugging
        self.__logger__.debug(f"Initializing...\nParameters: {dumps(obj = locals(), indent = 2, default = str)}")
    
        # Initialie model data record
        self._model_data_:      dict =          {}

        # Initialize distribution parameters
        self._kernel_:          str =           kernel
        self._kernel_group_:    int =           kernel_group
        self._locations_:       list[float] =   [location]*5
        self._scales_:          list[float] =   [scale]*5

        # Convolving layers
        self._conv1_:           Conv2d =        Conv2d(
                                                    in_channels =   channels_in, 
                                                    out_channels =  32, 
                                                    kernel_size =   3, 
                                                    padding =       1
                                                )
        
        self._conv2_:           Conv2d =        Conv2d(
                                                    in_channels =   32,
                                                    out_channels =  64,
                                                    kernel_size =   3,
                                                    padding =       1
                                                )
        
        self._conv3_:           Conv2d =        Conv2d(
                                                    in_channels =   64,
                                                    out_channels =  128,
                                                    kernel_size =   3,
                                                    padding =       1
                                                )
        
        self._conv4_:           Conv2d =        Conv2d(
                                                    in_channels =   128, 
                                                    out_channels =  256,
                                                    kernel_size =   3,
                                                    padding =       1
                                                )

        # Max pooling layers
        self._pool1_:           MaxPool2d =     MaxPool2d(
                                                    kernel_size =   2,
                                                    stride =        2
                                                )
        
        self._pool2_:           MaxPool2d =     MaxPool2d(
                                                    kernel_size =   2,
                                                    stride =        2
                                                )
        
        self._pool3_:           MaxPool2d =     MaxPool2d(
                                                    kernel_size =   2,
                                                    stride =        2
                                                )
        
        self._pool4_:           MaxPool2d =     MaxPool2d(
                                                    kernel_size =   2,
                                                    stride =        2
                                                )

        # FC layer
        self._fc_:              Linear =        Linear(
                                                    in_features =   dim ** 2,
                                                    out_features =  1024
                                                )

        # Classifier
        self._classifier_:      Linear =        Linear(
                                                    in_features =   1024,
                                                    out_features =  channels_out
                                                )

    def forward(self,
        X:  Tensor
    ) -> Tensor:
        """# Feed input through network and produce output.

        ## Args:
            * X (Tensor):   Input tensor.

        ## Returns:
            * Tensor:   Output tensor.
        """
        # INPUT LAYER =============================================================================
        # Log input tensor shape for debugging
        self.__logger__.debug(f"Input shape: {X.shape}")

        # Without taking gradients...
        with no_grad():
            
            # Convert tensor to float values
            y:  Tensor =    X.float()
            
            # Calculate mean & standard deviation of layer output
            self._locations_[0], self._scales_[0] = mean(y).item(), std(y).item()

        # LAYER 1 =================================================================================
        # Pass through first convolving layer
        x1:     Tensor =    self._conv1_(X)

        # Log first layer output for debugging
        self.__logger__.debug(f"Layer 1 output shape: {x1.shape}")

        # Without taking gradients...
        with no_grad(): 
            
            # Convert tensor to float values
            y:  Tensor =    x1.float()
            
            # Calculate mean & standard deviation of layer output
            self._locations_[1], self._scales_[1] = mean(y).item(), std(y).item()

        # LAYER 2 =================================================================================
        # Pass through second convolving layer
        x2:     Tensor =    self._conv2_(relu(self._pool1_(self._kernel1_(x1) if self._kernel_ is not None else x1)))

        # Log second layer output for debugging
        self.__logger__.debug(f"Layer 2 output shape: {x2.shape}")

        # Without taking gradients...
        with no_grad(): 
            
            # Convert tensor to float values
            y:  Tensor =    x2.float()
            
            # Calculate mean & standard deviation of layer output
            self._locations_[2], self._scales_[2] = mean(y).item(), std(y).item()

        # LAYER 3 =================================================================================
        # Pass through third convolving layer
        x3:     Tensor =    self._conv3_(relu(self._pool2_(self._kernel2_(x2) if self._kernel_ is not None else x2)))

        # Log third layer output for debugging
        self.__logger__.debug(f"Layer 3 output shape: {x3.shape}")

        # Without taking gradients...
        with no_grad(): 
            
            # Convert tensor to float values
            y:  Tensor =    x3.float()
            
            # Calculate mean & standard deviation of layer output
            self._locations_[3], self._scales_[3] = mean(y).item(), std(y).item()

        # LAYER 4 =================================================================================
        # Pass through third convolving layer
        x4:     Tensor =    self._conv4_(relu(self._pool3_(self._kernel3_(x3) if self._kernel_ is not None else x3)))

        # Log third layer output for debugging
        self.__logger__.debug(f"Layer 4 shape: {x4.shape}")

        # Without taking gradients...
        with no_grad(): 
            
            # Convert tensor to float values
            y:  Tensor =    x4.float()
            
            # Calculate mean & standard deviation of layer output
            self._locations_[4], self._scales_[4] = mean(y).item(), std(y).item()

        # OUTPUT LAYER ============================================================================
        output: Tensor =    relu(self._pool4_(self._kernel4_(x4) if self._kernel_ is not None else x4))

        self.__logger__.debug(f"Output shape: {output.shape}")

        # Return classified output
        return self._classifier_(relu(self._fc_(output.view(output.size(0), -1))))

    def _record_parameters_(self,
        epoch:  int
    ) -> None:
        """# Record mean & standard deviation of layers in model data file.
        
        ## Args:
            * epoch (int):  Epoch at which parameters are being recorded.
        """
        # Record epoch parameters
        self._model_data_.update({
            f"epoch-{epoch}":   {
                "locations": {f"layer-{l}": self._locations_[l - 1] for l in range(1, len(self._locations_) + 1)},
                "scales":    {f"layer-{s}": self._locations_[s - 1] for s in range(1, len(self._scales_)    + 1)}
            }
        })

    def save_parameters(self, 
        output_path:    str
    ) -> None:
        """# Dump model layer data to CSV file.

        ## Args:
            * output_path   (str):  Path at which data file (CSV) will be written
        """
        # Log action
        self.__logger__.info(f"Saving model layer data to {output_path}")
        
        # Save model data to file
        dump(
            obj =       self._model_data_,
            fp =        open(file = f"{output_path}/model_parameters.json", mode = "w"),
            indent =    2,
            default =   str
        )
    
    def set_kernels(self,
        epoch:          int,
        kernel_size:    int =   3
    ) -> None:
        """# Create/update kernels.

        ## Args:
            * epoch         (int):              Epoch during which kernels are being set.
            * kernel_size   (int, optional):    Size with which kernels will be created.
        """
        # Log for debugging
        self.__logger__.debug(f"EPOCH {epoch} locations: {self._locations_}, scales: {self._scales_}")
        
        # Set current epoch
        self._epoch_:   int =   epoch

        # For each kernel needed...
        for kernel, channel_size, location, scale in zip(
            ["_kernel1_", "_kernel2_", "_kernel3_", "_kernel4_"],
            [32, 64, 128, 256],
            self._locations_,
            self._scales_
        ):
            # Set kernel
            self.__setattr__(name = kernel, value = load_kernel(
                kernel =        self._kernel_,
                kernel_group =  self._kernel_group_,
                kernel_size =   kernel_size,
                channels =      channel_size,
                location =      location,
                scale =         scale
            ))