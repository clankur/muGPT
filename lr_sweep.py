import time
from clearml import Task
import numpy as np
import hydra
import jax_extra
import subprocess
from dataclasses import dataclass
from typing import Optional, Callable, Iterable
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

def wait_for_status_with_callback(
        task,
        status=(Task.TaskStatusEnum.completed, Task.TaskStatusEnum.stopped, Task.TaskStatusEnum.closed),
        raise_on_status=(Task.TaskStatusEnum.failed,),
        check_interval_sec=60.,
        callback: Optional[Callable[[], bool]] = None
):
    # type: (Task, Iterable[Task.TaskStatusEnum], Optional[Iterable[Task.TaskStatusEnum]], float, Optional[Callable[[], bool]]) -> None
    stopped_status = list(status) + (list(raise_on_status) if raise_on_status else [])
    while task.status not in stopped_status:
        if callback and callback():
            break
        time.sleep(check_interval_sec)
        task.reload()
    if raise_on_status and task.status in raise_on_status:
        raise RuntimeError(f"Task {task.task_id} has status: {task.status}.")
    # make sure we have the latest Task object
    task.reload()

def lr_sweep(
    config_name,
    model_name,
    queue_name,
    template_task_id,
    start_lr=0.005400,
    max_lr=5e-2,
    iterations=5,
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
            name=f"{model_name}_lr:{learning_rate:.6f}",
        )
        prev_loss = None
        def check_loss():
            nonlocal prev_loss
            scalars = child_task.get_last_scalar_metrics()
            loss = scalars["loss"]["loss"]["y"]
            if prev_loss and prev_loss > loss:
                return True
            print(prev_loss, loss, best_loss)
            prev_loss = loss
            return False
        
        child_task.set_parameter("Hydra/training.learning_rate", learning_rate)
        print(f"training model with lr: {learning_rate}")
        for i in range(3):
            try:
                Task.enqueue(child_task.id, queue_name=queue_name)
                wait_for_status_with_callback(
                    task=child_task,
                    callback=check_loss,
                    check_interval_sec=120
                )
                break
            except RuntimeError as e:
                if i + 1 == 3:
                    raise e
                print(e)
                child_task = Task.clone(
                    source_task=child_task.id,
                    name=f"{model_name}_lr:{learning_rate:.6f}",
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

    lr_low, lr_high = best_lr / search_mult, best_lr * search_mult
    for j in range(iterations):
        log_lr_low, log_lr_high = np.log10([ lr_low, lr_high ])
        
        log_lr_mid = (log_lr_low + log_lr_high) / 2
        lr_mid = 10 ** log_lr_mid
        loss = get_loss(lr_mid)
        
        if get_loss(lr_low) < loss:
            lr_high = lr_mid
        elif get_loss(lr_high) < loss:
            lr_low = lr_mid
        else:
            # If midpoint is best, narrow the search range
            lr_low = 10 ** ((log_lr_low + log_lr_mid) / 2)
            lr_high = 10 ** ((log_lr_high + log_lr_mid) / 2)
        logger.report_scalar("loss", "value", loss, iteration=i+j)

        print(f"Bounds = [{lr_low:.6f}, {lr_high:.6f}]")
        print(f"Iteration {i+j}: LR = {lr_mid:.6f}, Loss = {loss:.6f}")
        
    print(f"\nBest learning rate found: {best_lr:.6f} with loss: {best_loss:.6f}")

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
