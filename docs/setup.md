# Setup & Configuration
[< Documentation](./README.md)

##  Prerequisites

### Python
If you do not already have Python 3.10 or higher installed, do so [here](https://www.python.org/downloads/).

### Anaconda
If you do not already have Anaconda installed, you can find out how to do so [here](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).

### Create Conda Environment
Within the `conf/` directory, there is already an environment defined with all requirements. To create/activate it, simply run:
```
conda env create -f conf/adaptive-kernel_env.yaml   # Creation
conda activate hilo                                 # Activation
```

## Dependencies
In order to run the code within this package, one must have the necessary libraries/packages installed. If you have chosen not to utilize the Anaconda environment descrived in the [prerequisites](#create-conda-environment) section, then you will need to install the package dependencies by running the following command *from the project directory*:

```bash
pip install .
```