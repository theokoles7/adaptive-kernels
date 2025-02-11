"""Run series of jobs according to specifications."""

__all__ = ["run_experiment"]

from json               import dump, dumps
from logging            import Logger

from commands.run_job   import run_job

from utils              import LOGGER, TIMESTAMP

def run_experiment(
    datasets:           list[str],
    models:             list[str],
    kernels:            list[str] = [],
    kernel_groups:      list[int] = 13,
    kernel_size:        int =       3,
    location:           float =     0.0,
    scale:              float =     1.0,
    batch_size:         int =       64,
    data_path:          str =       "data",
    learning_rate:      float =     1e-1,
    epochs:             int =       200,
    output_path:        str =       "output",
    **kwargs
) -> None:
    """# Run series of jobs according to specifications.

    ## Dataset Args:
        * datasets          (list[str]):            Dataset on which job will execute.
        * batch_size        (int, optional):        Dataset batch size. Defaults to 64.
        * data_path         (str, optional):        Path at which dataset will be downloaded/loaded. 
                                                    Defaults to "./data/".
        
    ## Model Args:
        * models            (list[str]):            Model with which job will execute.
        * learning_rate     (float, optional):      Model's optimizer learning rate. Defaults to 
                                                    0.1.
        
    ## Kernel Args
        * kernels           (list[str], optional):  Kernel with which model will be loaded.
        * kernel_groups     (list[int], optional):  Kernel configuration group. Defaults to 13.
        * kernel_size       (int, optional):        Kernel size (square). Defaults to 3.
        * location          (float, optional):      Distribution location parameter. Defaults to 
                                                    0.0.
        * scale             (float, optional):      Distribution scale parameter. Defaults to 1.0.
        
    ## Job Args
        * epochs            (int, optional):        Number of epochs for which model will be 
                                                    trained. Defaults to 200.
        * output_path       (str, optional):        Path at which job results will be written. 
                                                    Defaults to "output".
    """
    # Initialize logger
    __logger__:                 Logger =    LOGGER.getChild(suffix = "experiment-process")
    
    # Log job configuration for debugging
    __logger__.debug(f"Initializing...\nParameters: {dumps(obj = locals(), indent = 2, default = str)}")
    
    # Calculate number of jobs being run
    _total_jobs_:               int =       (   # Jobs with kernels
                                                (
                                                    (len(datasets) * len(models) * len(kernels) * len(kernel_groups)) if kernels is not None else 0
                                                ) +
                                                # Jobs without kernels (control groups)
                                                (
                                                    len(datasets) * len(models)
                                                )
                                            )
    
    # Initialize experiment statistics
    _experiment_statistics_:    dict =      {
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
    
    # Initialize job IDs
    _job_id_:                   int =       1
    
    # Log amount of jobs being run
    __logger__.info(f"Running {_total_jobs_} total jobs")
    
    # For each dataset...
    for dataset in datasets:
        
        # For each model...
        for model in models:
            
            # For each kernel...
            for kernel in kernels:
                
                # For each kernel group...
                for kernel_group in kernel_groups:
                    
                    # Log job being run
                    __logger__.info(f"Executing job {_job_id_}/{_total_jobs_} | Dataset: {dataset} | Model: {model} | Kernel: {kernel} | Kernel Group: {kernel_group}")
                    
                    # Execute kernel job
                    job_statistics: dict =  run_job(
                                                dataset =       dataset,
                                                model =         model,
                                                kernel =        kernel,
                                                kernel_group =  kernel_group,
                                                kernel_size =   kernel_size,
                                                location =      location,
                                                scale =         scale,
                                                batch_size =    batch_size,
                                                data_path =     data_path,
                                                learning_rate = learning_rate,
                                                epochs =        epochs,
                                                output_path =   f"{output_path}/experiments/{TIMESTAMP}"
                                            )
                    
                    # Record job statistics
                    _experiment_statistics_["jobs"].update({_job_id_: {
                                                        "dataset":          dataset,
                                                        "model":            model,
                                                        "kernel":           kernel,
                                                        "kernel-group":     kernel_group,
                                                        "test-accuracy":    job_statistics["test_accuracy"],
                                                        "test-loss":        job_statistics["test_loss"]
                                                    }})
                    
                    # If new record is achieved on dataset...
                    if job_statistics["test_accuracy"] > _experiment_statistics_["dataset_performance_records"][dataset]["accuracy"]:
                        
                        # Record new record set
                        _experiment_statistics_["dataset_performance_records"][dataset].update({
                            "model":        model,
                            "kernel":       kernel,
                            "kernel-group": kernel_group,
                            "accuracy":     job_statistics["test_accuracy"],
                            "loss":         job_statistics["test_loss"]
                        })
                    
                    # Increment job ID
                    _job_id_ += 1
                    
            # Log job being run
            __logger__.info(f"Executing job {_job_id_}/{_total_jobs_} | Dataset: {dataset} | Model: {model} (Control Experiment)")
                    
            # Execute control job
            run_job(
                dataset =       dataset,
                model =         model,
                kernel_size =   kernel_size,
                location =      location,
                scale =         scale,
                batch_size =    batch_size,
                data_path =     data_path,
                learning_rate = learning_rate,
                epochs =        epochs,
                output_path =   f"{output_path}/experiments/{TIMESTAMP}"
            )
                    
            # Record job statistics
            _experiment_statistics_["jobs"].update({_job_id_: {
                                                "dataset":          dataset,
                                                "model":            model,
                                                "kernel":           "control",
                                                "kernel-group":     "control",
                                                "test-accuracy":    job_statistics["test_accuracy"],
                                                "test-loss":        job_statistics["test_loss"]
                                            }})
                    
            # If new record is achieved on dataset...
            if job_statistics["test_accuracy"] > _experiment_statistics_["dataset_performance_records"][dataset]["accuracy"]:
                
                # Record new record set
                _experiment_statistics_["dataset_performance_records"][dataset].update({
                    "model":        model,
                    "kernel":       "control",
                    "kernel-group": "control",
                    "accuracy":     job_statistics["test_accuracy"],
                    "loss":         job_statistics["test_loss"]
                })
                    
            # Increment job ID
            _job_id_ += 1
    
    # Save job statistics to file
    with open(
        file =      f"{output_path}/experiments/{TIMESTAMP}/statistics.json", 
        mode =      "w", 
        encoding =  "utf-8"
    ) as file_out: dump(obj = _experiment_statistics_, fp = file_out, indent = 2, default = str)
    
    # Log experiment results
    __logger__.info(f"Experiment record achievements: {dumps(obj = _experiment_statistics_['dataset_performance_records'], indent = 2, default = str)}")