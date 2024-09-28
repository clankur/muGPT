from clearml import Task
import numpy as np
import hydra
import jax_extra
from train import Config
import subprocess
from dataclasses import dataclass
from typing import Optional, Callable, Dict
from datetime import datetime
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from typing import Dict, List, Tuple


class SweepConfig(Config):
    base_task_id: Optional[str] = None


def get_task_details(config: Config):
    git_branch_name = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    config_name = hydra.core.hydra_config.HydraConfig.get()[
        "job"]["config_name"]
    project_name = (
        config.project_name
        if config.project_name
        else f"{config_name}/{git_branch_name}"
    )

    task_name = config.model_name

    return project_name, task_name


def bayesian_sweep(
    config_name,
    model_name,
    queue_name,
    template_task_id,
    training_length
):
    project_name = f"{config_name}/lr_sweep"
    task_name = f"{model_name}_lr_sweep_{datetime.now().strftime('%Y%m%d_%H%M')}"
    parent_task = Task.init(project_name=project_name, task_name=task_name)
    logger = parent_task.get_logger()

    def exponential_moving_average(data, alpha=0.03):
        """
        Compute exponential moving average using vectorized operations.
        alpha = 1 - smoothing_factor
        So for 0.97 smoothing, alpha = 1 - 0.97 = 0.03
        """
        weights = (1 - alpha) ** np.arange(len(data))
        weights /= weights.sum()
        ema = np.convolve(data, weights, mode="full")[: len(data)]
        return ema

    def train(parameters: Dict, template_task_id):
        # Clone the template task and override the learning rate
        model_overrides = ''.join(
            [f"_{key}={value}" for k, v in parameters.items()])
        child_task: Task = Task.clone(
            source_task=template_task_id,
            name=f"{model_name}_{model_overrides}",
        )
        child_task.set_system_tags([])
        for key, value in parameters.items():
            child_task.set_parameter(key, value)
        print(f"training model {model_name} with overrides: {model_overrides}")
        Task.enqueue(child_task.id, queue_name=queue_name)
        child_task.wait_for_status(check_interval_sec=600)

        # Get the loss from the child task
        scalars = child_task.get_reported_scalars()

        loss = scalars["loss"]["loss"]["y"]
        smoothed_loss = exponential_moving_average(loss, alpha=1 - 0.97)
        return smoothed_loss[-1], child_task.id

    # Define the search space
    space = [
        Real(3.0, 5.0, name='Hydra/training.amplitude'),
        Real(-0.6, -0.4, name='Hydra/training.power_law_exp'),
    ]

    @use_named_args(space)
    def objective(**params):
        # This wrapper allows skopt to use your train function
        params["Hydra/training.steps"] = current_training_length
        iteration = parent_task.get_last_iteration() + 1
        loss = train(params, template_task_id=template_task_id)
        for metric, value in params.items():
            title = f"{metric}_steps={current_training_length}"
            logger.report_scalar(
                title=title, series=metric, value=float(value), iteration=iteration)
            print('Best {}: {}'.format(title, value))

        return loss

    def bayesian_optimization(training_length: int, n_calls=50):
        """
        Perform Bayesian Optimization for hyperparameter tuning.

        :param n_calls: Number of iterations for the optimization
        :return: Result object from gp_minimize
        """
        global current_training_length
        current_training_length = training_length

        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=n_calls,
            n_random_starts=10,
            random_state=42,
            verbose=True
        )
        return result

    def print_result(result):
        """Print the best parameters and score."""
        print("Best parameters:")
        for name, value in zip([dim.name for dim in space], result.x):
            print(f"{name}: {value}")
        print(f"Best score: {result.fun}")

    def optimize_for_multiple_lengths(base_length: int, multipliers: List[int] = [1, 2, 4, 8], n_calls=50) -> List[Tuple[int, Dict, float]]:
        """
        Run Bayesian Optimization for multiple training lengths.

        :param base_length: The base training length
        :param multipliers: List of multipliers for the base length
        :param n_calls: Number of iterations for each optimization
        :return: List of tuples (training_length, best_params, best_score)
        """
        results = []

        for multiplier in multipliers:
            training_length = base_length * multiplier
            print(f"\nOptimizing for training length: {training_length}")

            result = bayesian_optimization(training_length, n_calls)

            best_params = dict(zip([dim.name for dim in space], result.x))
            results.append((training_length, best_params, result.fun))

        return results

    result = optimize_for_multiple_lengths(base_length=training_length)
    print_result(result)
    parent_task.close()


@hydra.main(config_path="configs", version_base=None)
def main(config):
    config = jax_extra.make_dataclass_from_dict(SweepConfig, config)
    config_name = hydra.core.hydra_config.HydraConfig.get()[
        "job"]["config_name"]
    project_name, task_name = get_task_details(config)

    print(f"{project_name=}")
    print(f"{task_name=}")

    template_task_id = config.base_task_id if config.base_task_id else Task.get_task(
        project_name=project_name,
        task_name=task_name,
    ).id

    print(f"{template_task_id=}")
    bayesian_sweep(
        config_name=config_name,
        model_name=config.model_name,
        queue_name=config.queue_name,
        template_task_id=template_task_id,
        training_length=config.training.steps
    )


if __name__ == "__main__":
    main()
