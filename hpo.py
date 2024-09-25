import sys
import subprocess
import argparse
from clearml import Task
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna
from clearml.automation.parameters import LogUniformParameterRange, UniformParameterRange


def create_optimizer(base_task_id: str, config: dict):
    return HyperParameterOptimizer(
        base_task_id=base_task_id,
        hyper_parameters=[
            UniformParameterRange(
                "Hydra/training.amplitude", min_value=3.0, max_value=5.0),
            UniformParameterRange(
                "Hydra/training.power_law_exp", min_value=-.6, max_value=-.4),
        ],
        objective_metric_title="loss",
        objective_metric_series="loss",
        objective_metric_sign="min_global",
        optimizer_class=OptimizerOptuna,
        execution_queue=config['queue'],
        max_number_of_concurrent_tasks=1,  # 100 in the paper
        total_max_jobs=25,  # 800 in the paper
        min_iteration_per_job=1,
        max_iteration_per_job=config["steps"],
    )


def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value,
          objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('WOOT WOOT we broke the record! Objective reached {}'.format(
            objective_value))


def main():
    parser = argparse.ArgumentParser(
        description='Run optimization with a specified base task ID.')
    parser.add_argument('--task_id', type=str,
                        help='The base task ID (required)')
    parser.add_argument('--queue', type=str, default="v4-32",
                        help='ClearML queue to run your tasks for HPO on')
    args = parser.parse_args()

    if args.task_id is None:
        parser.print_help()
        sys.exit(1)

    base_task_id = args.task_id
    base_task: Task = Task.get_task(base_task_id)
    project_name, task_name = base_task.get_project_name(), base_task.name
    config = base_task.get_configuration_object_as_dict('OmegaConf')
    config['queue'] = args.queue
    git_branch_name = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()

    print(f"Using task ID: {base_task_id}")
    print(f"Project name: {project_name}, Task name: {task_name}")
    task: Task = Task.init(
        project_name=f"{project_name}/hpo",
        task_name=f'hpo_{task_name}_{git_branch_name}_search',
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False
    )
    optim = create_optimizer(base_task_id, config)
    # report every 22.5 mins
    optim.set_report_period(22.50)
    # start the optimization process, callback function to be called every time an experiment is completed
    # this function returns immediately
    optim.start(job_complete_callback=job_complete_callback)
    # wait until process is done (notice we are controlling the optimization process in the background)
    optim.wait()
    # optimization is completed, print the top performing experiments id
    top_exp = optim.get_top_experiments(top_k=10)
    top_details = optim.get_top_experiments_details(top_k=10)
    print([t.id for t in top_exp])
    print([t for t in top_details])
    # make sure background optimization stopped
    optim.stop()


if __name__ == "__main__":
    main()
