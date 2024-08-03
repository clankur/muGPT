import time
from clearml import Task
import numpy as np
import hydra
import jax_extra
import subprocess
from dataclasses import dataclass
from typing import Optional, Callable, Iterable
from datetime import datetime
import json


@dataclass
class Config:
    model_name: str
    queue_name: str
    project_name: Optional[str] = None


def get_task_details(config: Config):
    git_branch_name = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    config_name = hydra.core.hydra_config.HydraConfig.get()["job"]["config_name"]
    project_name = (
        config.project_name
        if config.project_name
        else f"{config_name}/{git_branch_name}"
    )

    task_name = config.model_name

    return project_name, task_name


def lr_sweep(
    config_name,
    model_name,
    queue_name,
    template_task_id,
    start_lr=5e-4,
    max_lr=5e-2,
    iterations=5,
    search_mult=3,
):
    project_name = f"{config_name}/lr_sweep"
    task_name = f"{model_name}_lr_sweep_{datetime.now().strftime('%Y%m%d_%H%M')}"
    parent_task = Task.init(project_name=project_name, task_name=task_name)
    logger = parent_task.get_logger()
    loss_per_learning_rate = {}
    i = 0

    best_lr, best_loss = None, float("inf")

    def exponential_search():
        nonlocal i, best_lr, best_loss
        current_lr = start_lr

        while current_lr <= max_lr:
            current_loss = get_loss(current_lr)
            if current_loss < best_loss:
                best_lr, best_loss = current_lr, current_loss
            else:
                break
            print(f"Iteration {i}: LR = {current_lr:.6f}, Loss = {current_loss:.6f}")
            logger.report_scalar("loss", "value", current_loss, iteration=i)
            logger.report_scalar("lr", "value", current_lr, iteration=i)

            logger.report_scalar("loss", "best", best_loss, iteration=i)
            logger.report_scalar("lr", "best", best_lr, iteration=i)

            i += 1
            current_lr *= search_mult

        return best_lr / search_mult, best_lr * search_mult

    def binary_search(lr_low, lr_high):
        nonlocal best_loss, best_lr

        for j in range(1, iterations + 1):
            print(f"Bounds = [{lr_low:.6f}, {lr_high:.6f}]")
            log_lr_low, log_lr_high = np.log10([lr_low, lr_high])

            log_lr_mid = (log_lr_low + log_lr_high) / 2
            lr_mid = 10**log_lr_mid
            loss, low_loss, high_loss = (
                get_loss(lr_mid),
                get_loss(lr_low),
                get_loss(lr_high),
            )

            for lr, loss in [(lr_low, low_loss), (lr_mid, loss), (lr_high, high_loss)]:
                if loss < best_loss:
                    best_loss, best_lr = loss, lr

            logger.report_scalar("loss", "best", best_loss, iteration=i + j)
            logger.report_scalar("lr", "best", best_lr, iteration=i + j)

            logger.report_scalar("loss", "upper", high_loss, iteration=i + j)
            logger.report_scalar("loss", "value", loss, iteration=i + j)
            logger.report_scalar("loss", "lower", low_loss, iteration=i + j)
            logger.report_scalar("lr", "upper", lr_high, iteration=i + j)
            logger.report_scalar("lr", "value", lr_mid, iteration=i + j)
            logger.report_scalar("lr", "lower", lr_low, iteration=i + j)

            print(f"Iteration {i+j}: LR = {lr_mid:.6f}, Loss = {loss:.6f}")

            if low_loss < loss:
                lr_high = lr_mid
            elif high_loss < loss:
                lr_low = lr_mid
            else:
                # If midpoint is best, narrow the search range
                lr_low = 10 ** ((log_lr_low + log_lr_mid) / 2)
                lr_high = 10 ** ((log_lr_high + log_lr_mid) / 2)

        return best_lr

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

    def train(learning_rate, template_task_id):
        # Clone the template task and override the learning rate
        child_task: Task = Task.clone(
            source_task=template_task_id,
            name=f"{model_name}_lr:{learning_rate:.6f}",
        )
        child_task.set_system_tags([])
        child_task.set_parameter("Hydra/training.learning_rate", learning_rate)
        print(f"training model with lr: {learning_rate}")
        Task.enqueue(child_task.id, queue_name=queue_name)
        child_task.wait_for_status(check_interval_sec=120)

        # Get the loss from the child task
        scalars = child_task.get_reported_scalars()

        loss = scalars["loss"]["loss"]["y"]
        smoothed_loss = exponential_moving_average(loss, alpha=1 - 0.97)
        return smoothed_loss[-1], child_task.id

    def get_loss(lr):
        lr = round(lr, 5)
        if lr not in loss_per_learning_rate:
            loss_per_learning_rate[lr] = train(lr, template_task_id)
        return loss_per_learning_rate[lr][0]

    lr_low, lr_high = exponential_search()
    print("proceeding with binary search now")
    best_lr = binary_search(lr_low, lr_high)

    print(f"\nBest learning rate found: {best_lr:.6f} with loss: {best_loss:.6f}")
    print(f"all experiments run: {json.dumps(loss_per_learning_rate)}")

    parent_task.close()
    return best_lr


@hydra.main(config_path="configs", version_base=None)
def main(config):
    config = jax_extra.make_dataclass_from_dict(Config, config)
    config_name = hydra.core.hydra_config.HydraConfig.get()["job"]["config_name"]
    project_name, task_name = get_task_details(config)

    print(f"{project_name=}")
    print(f"{task_name=}")

    template_task_id = Task.get_task(
        project_name=project_name,
        task_name=task_name,
    ).id

    lr_sweep(
        config_name=config_name,
        model_name=config.model_name,
        queue_name=config.queue_name,
        template_task_id=template_task_id,
    )


if __name__ == "__main__":
    main()
