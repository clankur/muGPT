from clearml import Task
import numpy as np
import hydra
import jax_extra
from train import Config
import subprocess
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


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
    start_lr=1e-5,
    max_lr=5e-2,
    search_mult=3,
):
    project_name = f"{config_name}/lr_sweep"
    task_name = f"{model_name}_lr_sweep_{datetime.now().strftime('%Y%m%d_%H%M')}"
    parent_task = Task.init(project_name=project_name, task_name=task_name)
    logger = parent_task.get_logger()
    loss_per_learning_rate = {}

    def train(learning_rate, template_task_id):
        # Clone the template task and override the learning rate
        child_task = Task.clone(
            source_task=template_task_id,
            name=f"{task_name}_lr:{learning_rate:.6f}",
        )
        child_task.set_parameter("Hydra/training.learning_rate", learning_rate)
        print(f"training model with lr: {learning_rate}")
        for i in range(3):
            try:
                Task.enqueue(child_task.id, queue_name=queue_name)
                child_task.wait_for_status()
                break
            except RuntimeError as e:
                if i + 1 == 3:
                    raise e
                print(e)
                child_task = Task.clone(
                    source_task=child_task.id,
                    name=f"{task_name}_lr:{learning_rate:.6f}",
                )

        # Get the loss from the child task
        child_task_results = child_task.get_reported_scalars()

        return child_task_results["loss"]["loss"]["y"][-1]

    def get_loss(lr):
        if lr not in loss_per_learning_rate:
            loss_per_learning_rate[lr] = train(lr, template_task_id)
        return loss_per_learning_rate[lr]

    i = 0
    current_lr = start_lr
    best_lr, best_loss = current_lr, get_loss(current_lr)
    logger.report_scalar("loss", "value", best_loss, iteration=i)
    while current_lr <= max_lr:
        i += 1
        current_lr *= search_mult
        current_loss = get_loss(current_lr)
        print(f"Iteration {i+1}: LR = {current_lr:.6f}, Loss = {current_loss:.6f}")
        logger.report_scalar("loss", "value", current_loss, iteration=i)
        if current_loss < best_loss:
            best_lr, best_loss = current_lr, current_loss
        else:
            break
    
    print("proceeding with binary search now")
    
    lower_bound, upper_bound = np.log10([best_lr, current_lr])
    while 10**upper_bound - 10**lower_bound > 1e5:
        midpoint = (lower_bound + upper_bound) / 2
        loss = get_loss(10**midpoint)
        i += 1
        if loss < best_loss:
            best_loss, best_lr = loss, 10**midpoint
            lower_bound = midpoint
        else:
            upper_bound = midpoint

        logger.report_scalar("loss", "value", loss, iteration=i)

        print(f"Bounds = [{10**lower_bound:.6f}, {10**upper_bound:.6f}]")
        print(f"Iteration {i+1}: LR = {midpoint:.6f}, Loss = {loss:.6f}")

    print(f"\nBest learning rate found: {best_lr:.6f} with loss: {best_loss:.6f}")

    parent_task.close()


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
