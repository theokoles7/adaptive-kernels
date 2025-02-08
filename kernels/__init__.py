"""Kernels package."""

__all__ = ["__base__", "cauchy", "gaussian", "gumbel", "laplace", "load_kernel"]

from kernels.__base__           import Kernel

from kernels.cauchy             import CauchyKernel
from kernels.gaussian           import GaussianKernel
from kernels.gumbel             import GumbelKernel
from kernels.laplace            import LaplaceKernel

def load_kernel(
    kernel:         str,
    kernel_group:   int =   13,
    kernel_size:    int =   3,
    location:       float = 0.0,
    scale:          float = 1.0,
    channels:       int =   3,
    **kwargs
) -> Kernel:
    """# Load specified kernel.

    ## Args:
        * kernel        (str):              Kernel selection.
        * kernel_group  (int):              Kernel configuration type. Defaults to 13.
        * kernel_size   (int, optional):    Kernel size (square). Defaults to 3.
        * location      (float, optional):  Distribution location parameter.
        * scale         (float, optional):  Distribution scale parameter.
        * channels      (int, optional):    Input channels. Defaults to 3.

    ## Returns:
        * Kernel:   Selected kernel, initialized and ready for convolving.
    """
    # Validate kernel group selection
    assert kernel_group in range(1, 15),    f"Invalid kernel group selection: {kernel_group}"
    
    # Match kernel selection
    match kernel:
        
        # Cauchy
        case "cauchy":      return CauchyKernel(**locals())
        
        # Gaussian
        case "gaussian":    return GaussianKernel(**locals())
        
        # Gumbel
        case "gumbel":      return GumbelKernel(**locals())
        
        # Laplace
        case "laplace":     return LaplaceKernel(**locals())
        
        # Invalid selection
        case _:             raise NotImplementedError(f"Invalid kernel selection: {kernel}")