from clearml.automation import UniformParameterRange, UniformIntegerParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna
from clearml import Task
import numpy as np
import hydra
import jax
import jax_extra
import os
from train import clear_locks, Config
import subprocess

initial_lr_range_log = (np.log10(1e-5), np.log10(1e-1))  # Initial learning rate range in log10 scale

def create_hpo_task(lr_range_log, queue):
    return HyperParameterOptimizer(
        base_task_id='2fbbd3a9216d4e9c963b48b70fcebd89',  # 270m base task id 
        hyper_parameters=[
            UniformParameterRange('Hydra/training.learning_rate', 10**lr_range_log[0], 10**lr_range_log[1]),
        ],
        objective_metric_title='validation_accuracy',  # Metric to optimize
        objective_metric_series='Series',  # Specific metric series
        max_number_of_concurrent_tasks=2,
        optimizer_class=OptimizerOptuna,
        total_max_jobs=10,
        execution_queue=queue,
        min_iteration_per_job=10,
        max_iteration_per_job=30,
    )

def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('WOOT WOOT we broke the record! Objective reached {}'.format(objective_value))

@hydra.main(config_path='configs', version_base=None)
def main(config):
  config = jax_extra.make_dataclass_from_dict(Config, config)
  if config.training.queue:
    # config_name = hydra.core.hydra_config.HydraConfig.get()['job']['config_name']
    git_branch_name = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.strip()

    task = Task.init(
        project_name="HPO",
        task_name=f"Learning rate search {config.paths.model_name}-{git_branch_name}",
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False
    )
    
    an_optimizer = create_hpo_task(initial_lr_range_log, config.training.queue)
    # report every 12 seconds, this is way too often, but we are testing here J
    an_optimizer.set_report_period(0.2)
    # start the optimization process, callback function to be called every time an experiment is completed
    # this function returns immediately
    an_optimizer.start(job_complete_callback=job_complete_callback)
    # set the time limit for the optimization process (2 hours)

    # set the time limit for the optimization process (2 hours)
    an_optimizer.set_time_limit(in_minutes=90.0)
    # wait until process is done (notice we are controlling the optimization process in the background)
    an_optimizer.wait()
    # optimization is completed, print the top performing experiments id
    top_exp = an_optimizer.get_top_experiments(top_k=3)
    print([t.id for t in top_exp])
    # make sure background optimization stopped
    an_optimizer.stop()

    print('We are done, good bye')

    # Step 4: Finalize
    task.close()
    


if __name__ == "__main__":
  main()


# Step 2: Define the initial search space in log scale

# Function to create a new HPO task with a specified learning rate range
