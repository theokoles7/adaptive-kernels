"""Models package."""

__all__ = ["load_model", "normal", "resnet_block", "resnet", "vgg"]

from torch.nn               import Module

from models.normal          import NormalCNN
from models.resnet          import Resnet
from models.vgg             import VGG

# Define Model accessor
def load_model(
    model:          str,
    channels_in:    int, 
    channels_out:   int, 
    dim:            int,
    kernel:         str =   None,
    kernel_group:   int =   13,
    location:       float = 0.0,
    scale:          float = 1.0,
    **kwargs
) -> Module:
    """# Provide selected model.

    ## Args:
        * model         (str):              Choice of model (normal, resnet, vgg).
        * channels_in   (int):              Input channels
        * channels_out  (int):              Output channels
        * dim           (int):              Input image dimension
        * kernel        (str, optional):    Kernel with which model will be initialized.
        * kernel_group  (int, optional):    Kernel configuration type. Defaults to 13.
        * location      (float, optional):  Distribution location parameter. Defaults to 0.0.
        * scale         (float, optional):  Distribution scale parameter. Defaults to 1.0.

    ## Returns:
        * Module:   Selected model
    """
    # Match model selection
    match model:
        
        # Custom CNN
        case "normal-cnn":  return NormalCNN(**locals())
        
        # Resnet 18
        case "resnet":      return Resnet(**locals())
        
        # VGG 16
        case "vgg":         return VGG(**locals())

        # Report invalid model selection
        case _:             raise NotImplementedError(f"{model} not supported.")