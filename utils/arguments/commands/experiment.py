"""Experiment argument definitions."""

__all__ = ["add_experiment_parser"]

from argparse                   import _ArgumentGroup, ArgumentParser, _SubParsersAction

def add_experiment_parser(
    parent_subparser:   _SubParsersAction
) -> None:
    """# Add parser/arguments for running an experiment (job series).

    ## Args:
        * parent_subparser  (_SubParsersAction):    Parent's sub-parser.
    """
    # Inialize parser
    _parser_:       ArgumentParser =    parent_subparser.add_parser(
        name =      "run-experiment",
        help =      """Execute an experiment process (job series)."""
    )
    
    # Add experiment argument(s)
    _parser_.add_argument(
        "--epochs",
        type =      int,
        default =   200,
        help =      """Number of epochs for which training phase of job will execute. Defaults to 
                    200."""
    )
    
    # DATASETS =====================================================================================
    # Initialize datasets arguments group
    _datasets_:     _ArgumentGroup =    _parser_.add_argument_group("Datasets")
    
    # Add experiment arguments
    _datasets_.add_argument(
        "--datasets",
        type =      str,
        nargs =     "+",
        choices =   ["cifar10", "cifar100", "imagenet", "mnist"],
        help =      """Dataset(s) on which experiments will be executed."""
    )
    
    # Data path
    _datasets_.add_argument(
        "--data-path",
        type =      str,
        default =   "data",
        help =      """Path at which dataset will be downloaded/loaded. Defaults to "./data/"."""
    )
    
    # Batch size
    _datasets_.add_argument(
        "--batch-size",
        type =      int,
        default =   64,
        help =      """Dataset batch size for training phase. Defaults to 64."""
    )
    
    # MODELS =======================================================================================
    # Initialize models arguments group
    _models_:       _ArgumentGroup =    _parser_.add_argument_group("Models")
    
    _models_.add_argument(
        "--models",
        type =      str,
        nargs =     "+",
        choices =   ["normal-cnn", "resnet", "vgg"],
        help =      """Model(s) on which experiments will be executed."""
    )
    
    # Add model arguments
    _models_.add_argument(
        "--learning-rate", "-lr",
        type =      float,
        default =   1e-1,
        help =      """Model's learning rate. Defaults to 0.1."""
    )
    
    # KERNELS ======================================================================================
    # Initialize kernels arguments group
    _kernels_:      _ArgumentGroup =    _parser_.add_argument_group("Kernels")
    
    _kernels_.add_argument(
        "--kernels",
        type =      str,
        nargs =     "+",
        choices =   ["cauchy", "gaussian", "gumbel", "laplace"],
        default =   [],
        help =      """Kernel(s) on which experiments will be executed. Equivalent of experiment(s) 
                    with no kernel will automatically be executed for comparison."""
    )
    
    # Add Gaussian kernel arguments
    _kernels_.add_argument(
        "--location", "-mu",
        type =      float,
        default =   0.0,
        help =      """Location (Mu/Mean) parameter. Defaults to 0."""
    )
    
    _kernels_.add_argument(
        "--scale", "-sigma",
        type =      float,
        default =   1.0,
        help =      """Scale (Sigma/Variance) parameter. Defaults to 1."""
    )
    
    _kernels_.add_argument(
        "--kernel-size",
        type =      int,
        default =   3,
        help =      """Kernel size (square). Defaults to 3."""
    )
    
    _kernels_.add_argument(
        "--kernel-groups",
        type =      int,
        choices =   range(1, 15),
        nargs =     "+",
        default =   [13,],
        help =      """Kernel configuration type. Defaults to 13."""
    )