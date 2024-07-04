from clearml.automation import LogUniformParameterRange, UniformIntegerParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna
from clearml import Task
import numpy as np
import hydra
import jax
import jax_extra
import os
from train import Config
import subprocess

args = {
    "template_task_id": '69f514621af04bb69ec5e8e84b948635',
    "run_as_service": False,
}


def create_hpo_task(config: Config):
    if not args["template_task_id"]:
        git_branch_name = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        config_name = hydra.core.hydra_config.HydraConfig.get()["job"]["config_name"]
        project_name = f"{config_name}/{git_branch_name}"
        task_name = config.paths.model_name

        print(f"{project_name=}")
        print(f"{task_name=}")
        args["template_task_id"] = Task.get_task(
            project_name=f"{config_name}/{git_branch_name}",
            task_name=config.paths.model_name,
        ).id

    return HyperParameterOptimizer(
        # specifying the task to be optimized, task must be in system already so it can be cloned
        #   base_task_id=TEMPLATE_TASK_ID,
        base_task_id=args["template_task_id"],  # 2fbbd3a9216d4e9c963b48b70fcebd89
        # setting the hyperparameters to optimize
        hyper_parameters=[
            LogUniformParameterRange(
                "Hydra/training.learning_rate", min_value=-5, max_value=0
            )
        ],
        # setting the objective metric we want to maximize/minimize
        objective_metric_title="loss",
        objective_metric_series="loss",
        objective_metric_sign="min",
        # setting optimizer
        optimizer_class=OptimizerOptuna,
        # configuring optimization parameters
        execution_queue=config.training.queue,
        max_number_of_concurrent_tasks=1,
        #   optimization_time_limit=120.,
        #   compute_time_limit=1200,
        total_max_jobs=2,
        min_iteration_per_job=1,
        max_iteration_per_job=150000,
    )


def job_complete_callback(
    job_id,  # type: str
    objective_value,  # type: float
    objective_iteration,  # type: int
    job_parameters,  # type: dict
    top_performance_job_id,  # type: str
):
    print(
        "Job completed!", job_id, objective_value, objective_iteration, job_parameters
    )
    if job_id == top_performance_job_id:
        print(
            "WOOT WOOT we broke the record! Objective reached {}".format(
                objective_value
            )
        )


@hydra.main(config_path="configs", version_base=None)
def main(config):
    config = jax_extra.make_dataclass_from_dict(Config, config)
    if config.training.queue:
        # config_name = hydra.core.hydra_config.HydraConfig.get()['job']['config_name']
        git_branch_name = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()

        task = Task.init(
            project_name="HPO",
            task_name=f"Learning rate search {config.paths.model_name}",
            task_type=Task.TaskTypes.optimizer,
            reuse_last_task_id=False,
        )
        # task.execute_remotely(queue_name='seqax', exit_process=True)

        optimizer = create_hpo_task(config)
        # report every 12 seconds, this is way too often, but we are testing here J
        optimizer.set_report_period(30)
        # start the optimization process, callback function to be called every time an experiment is completed
        # this function returns immediately
        optimizer.start(job_complete_callback=job_complete_callback)

        # set the time limit for the optimization process (2 hours)
        optimizer.set_time_limit(in_minutes=120.0)
        # wait until process is done (notice we are controlling the optimization process in the background)
        optimizer.wait()
        # optimization is completed, print the top performing experiments id
        top_exp = optimizer.get_top_experiments(top_k=5)
        print([t.id for t in top_exp])
        # make sure background optimization stopped
        optimizer.stop()

        print("We are done, good bye")

        # Step 4: Finalize
        task.close()


if __name__ == "__main__":
    main()


# Step 2: Define the initial search space in log scale

# Function to create a new HPO task with a specified learning rate range
