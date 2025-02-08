"""VGG-16 model."""

__all__ = ["VGG"]

from json                   import dump, dumps
from logging        import Logger

from torch          import mean, no_grad, std, Tensor
from torch.nn       import BatchNorm2d, Conv2d, Dropout, Linear, MaxPool2d, Module, ReLU, Sequential
from torch.nn.init  import constant_, kaiming_normal_, normal_

from kernels        import load_kernel
from utils          import LOGGER

class VGG(Module):
    """VGG 16 model."""

    def __init__(self, 
        channels_in:    int, 
        channels_out:   int,
        kernel:         str =   None,
        kernel_group:   int =   13,
        location:       float = 0.0,
        scale:          float = 1.0,
        **kwargs
    ):
        """# Initialize VGG-16 model.

        ## Args:
            * channels_in   (int):              Input channels.
            * channels_out  (int):              Output channels.
            * kernel        (str, optional):    Kernel with which model will be set.
            * kernel_group  (int, optional):    Kernel configuration type. Defaults to 13.
            * location      (float, optional):  Distribution location parameter. Defaults to 0.0.
            * scale         (float, optional):  Distribution scale parameter. Defaults to 1.0.
        """
        # Initialize Module object
        super(VGG, self).__init__()

        # Initialize logger
        self.__logger__:        Logger =        LOGGER.getChild(suffix = 'vgg')
        
        # Log initialization for debugging
        self.__logger__.debug(f"Initializing...\nParameters: {dumps(obj = locals(), indent = 2, default = str)}")
    
        # Initialie model data record
        self._model_data_:      dict =          {}

        # Initialize distribution parameters
        self._kernel_:          str =           kernel
        self._kernel_group_:    int =           kernel_group
        self._locations_:       list[float] =   [location]*6
        self._scales_:          list[float] =   [scale]*6

        # Convolving layers
        self._conv1_:           Sequential =    Sequential(
                                                    Conv2d(
                                                        in_channels =   channels_in,
                                                        out_channels =  64,
                                                        kernel_size =   3,
                                                        padding =       1
                                                    ),
                                                    ReLU(),
                                                    Conv2d(
                                                        in_channels =   64,
                                                        out_channels =  64,
                                                        kernel_size =   3,
                                                        padding =       1
                                                    )
                                                )
        
        self._conv2_:           Sequential =    Sequential(
                                                    Conv2d(
                                                        in_channels =   64,
                                                        out_channels =  128,
                                                        kernel_size =   3,
                                                        padding =       1
                                                    ),
                                                    ReLU(),
                                                    Conv2d(
                                                        in_channels =   128,
                                                        out_channels =  128,
                                                        kernel_size =   3,
                                                        padding =       1
                                                    )
                                                )
        
        self._conv3_:           Sequential =    Sequential(
                                                    Conv2d(
                                                        in_channels =   128,
                                                        out_channels =  256,
                                                        kernel_size =   3,
                                                        padding =       1
                                                    ),
                                                    ReLU(),
                                                    Conv2d(
                                                        in_channels =   256,
                                                        out_channels =  256,
                                                        kernel_size =   3,
                                                        padding =       1
                                                    )
                                                )
        
        self._conv4_:           Sequential =    Sequential(
                                                    Conv2d(
                                                        in_channels =   256,
                                                        out_channels =  512,
                                                        kernel_size =   3,
                                                        padding =       1
                                                    ),
                                                    ReLU(),
                                                    Conv2d(
                                                        in_channels =   512,
                                                        out_channels =  512,
                                                        kernel_size =   3,
                                                        padding =       1
                                                    )
                                                )
        
        self._conv5_:           Sequential =    Sequential(
                                                    Conv2d(
                                                        in_channels =   512,
                                                        out_channels =  512,
                                                        kernel_size =   3,
                                                        padding =       1
                                                    ),
                                                    ReLU(),
                                                    Conv2d(
                                                        in_channels =   512,
                                                        out_channels =  512,
                                                        kernel_size =   3,
                                                        padding =       1
                                                    )
                                                )

        # Pooling layers
        self._pool1_:           Sequential =    Sequential(
                                                    ReLU(),
                                                    MaxPool2d(
                                                        kernel_size =   2,
                                                        stride =        2
                                                    )
                                                )
        self._pool2_:           Sequential =    Sequential(
                                                    ReLU(),
                                                    MaxPool2d(
                                                        kernel_size =   2,
                                                        stride =        2
                                                    )
                                                )
        self._pool3_:           Sequential =    Sequential(
                                                    ReLU(),
                                                    MaxPool2d(
                                                        kernel_size =   2,
                                                        stride =        2
                                                    )
                                                )
        self._pool4_:           Sequential =    Sequential(
                                                    ReLU(),
                                                    MaxPool2d(
                                                        kernel_size =   2,
                                                        stride =        2
                                                    )
                                                )
        self._pool5_:           Sequential =    Sequential(
                                                    ReLU(),
                                                    MaxPool2d(
                                                        kernel_size =   2,
                                                        stride =        2
                                                    )
                                                )

        # Classifier
        self.classifier:        Sequential =    Sequential(
                                                    Linear(
                                                        in_features =   512,
                                                        out_features =  4096
                                                    ), 
                                                    ReLU(), 
                                                    Dropout(),
                                                    Linear(
                                                        in_features =   4096,
                                                        out_features =  4096
                                                    ),
                                                    ReLU(),
                                                    Dropout(),
                                                    Linear(
                                                        in_features =   4096,
                                                        out_features =  channels_out
                                                    )
                                                )

        # Initialize layer weigthts
        self._initialize_weights_()

    def forward(self,
        X: Tensor
    ) -> Tensor:
        """# Feed input through network and provide output.

        ## Args:
            * X                     (Tensor):           Input tensor.
            * return_intermediate   (bool, optional):   If True, returns intermediate output, prior 
                                                        to classification. Defaults to False.

        ## Returns:
            * Tensor:   Output tensor.
        """
        # INPUT LAYER =============================================================================
        self.__logger__.debug(f"Input shape: {X.shape}")

        # Without taking gradients...
        with no_grad(): 
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
        x2:     Tensor =    self._conv2_(self._pool1_(self._kernel1_(x1) if self._kernel_ is not None else x1))

        self.__logger__.debug(f"Layer 2 output shape: {x2.shape}")

        # Without taking gradients...
        with no_grad(): 
            
            # Convert tensor to float values
            y:  Tensor =    x2.float()
            
            # Calculate mean & standard deviation of layer output
            self._locations_[2], self._scales_[2] = mean(y).item(), std(y).item()

        # LAYER 3 =================================================================================
        # Pass through third convolving layer
        x3:     Tensor =    self._conv3_(self._pool2_(self._kernel2_(x2) if self._kernel_ is not None else x2))

        self.__logger__.debug(f"Layer 3 output shape: {x3.shape}")

        # Without taking gradients...
        with no_grad(): 
            
            # Convert tensor to float values
            y:  Tensor =    x3.float()
            
            # Calculate mean & standard deviation of layer output
            self._locations_[3], self._scales_[3] = mean(y).item(), std(y).item()

        # LAYER 4 =================================================================================
        # Pass through fourth convolving layer
        x4:     Tensor =    self._conv4_(self._pool3_(self._kernel3_(x3) if self._kernel_ is not None else x3))

        self.__logger__.debug(f"Layer 4 output shape: {x4.shape}")

        # Without taking gradients...
        with no_grad(): 
            
            # Convert tensor to float values
            y:  Tensor =    x4.float()
            
            # Calculate mean & standard deviation of layer output
            self._locations_[4], self._scales_[4] = mean(y).item(), std(y).item()

        # LAYER 5 =================================================================================
        # Pass through fifth convolving layer
        x5:     Tensor =    self._conv5_(self._pool4_(self._kernel4_(x4) if self._kernel_ is not None else x4))

        self.__logger__.debug(f"Layer 5 output shape: {x5.shape}")

        # Without taking gradients...
        with no_grad(): 
            
            # Convert tensor to float values
            y:  Tensor =    x5.float()
            
            # Calculate mean & standard deviation of layer output
            self._locations_[5], self._scales_[5] = mean(y).item(), std(y).item()

        # OUTPUT LAYER ============================================================================
        # Pass through output convolving layer
        output: Tensor =    self._pool5_(self._kernel5_(x5) if self._kernel_ is not None else x5)
        
        # Return classification
        return self.classifier(output.view(output.size(0), -1))

    def _initialize_weights_(self) -> None:
        """Initialize weights in all layers."""
        # For each module...
        for module in self.modules():

            # For convolving layer(s)...
            if isinstance(module, Conv2d):
                
                # Fill the input Tensor with values using a Kaiming normal distribution
                kaiming_normal_(tensor = module.weight, mode = 'fan_in', nonlinearity = 'relu')
                if module.bias is not None: constant_(module.bias, 0)

            # For batch normalization
            elif isinstance(module, BatchNorm2d):
                
                # Fill the input Tensor with the value
                constant_(tensor = module.weight, val = 1)
                constant_(tensor = module.bias,   val = 0)

            # For linear layers
            elif isinstance(module, Linear):
                
                # Fill the input Tensor with values drawn from the normal distribution
                normal_(tensor = module.weight, mean = 0, std = 0.01)
                
                # Fill the input Tensor with the value
                constant_(tensor = module.bias, val = 0)

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
            * epoch         (int):              Current epoch number
            * kernel_size   (int, optional):    Kernel size (square). Defaults to 3.
        """
        # Log for debugging
        self.__logger__.debug(f"EPOCH {epoch} locations: {self._locations_}, scales: {self._scales_}")
        
        # Set current epoch
        self._epoch_:   int =   epoch

        # For each kernel needed...
        for kernel, channel_size, location, scale in zip(
            ["_kernel1_", "_kernel2_", "_kernel3_", "_kernel4_", "_kernel5_"],
            [64, 128, 256, 512, 512],
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