from clearml import Task
import numpy as np
import hydra
import jax_extra
from train import Config
import subprocess
from dataclasses import dataclass
from typing import Optional

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
    project_name = config.project_name if config.project_name else f"{config_name}/{git_branch_name}"
    task_name = config.model_name

    return project_name, task_name
    
def lr_sweep_binary_search(
    config: Config, lower_bound, upper_bound, template_task_id, max_iterations=5 
):
    config_name = hydra.core.hydra_config.HydraConfig.get()["job"]["config_name"]
    project_name = f"{config_name}/lr_sweep"
    task_name = f"{config.model_name}_lr_sweep"
    parent_task = Task.init(project_name=project_name, task_name=task_name)
    pt_logger = parent_task.get_logger()
    best_lr = None
    best_loss = float("inf")

    loss_per_learning_rate = {} 
    def train (learning_rate, template_task_id):
        # Clone the template task and override the learning rate
        child_task = Task.clone(
            source_task=template_task_id, name=f"{task_name}_lr:{learning_rate:.6f}"
        )
        child_task.set_parameter("Hydra/training.learning_rate", learning_rate ) 
        print(f"training model with lr: {learning_rate}")
        for i in range(3):
            try:
                Task.enqueue(child_task.id, queue_name=config.queue_name)
                child_task.wait_for_status()
                break
            except RuntimeError as e:
                if i + 1 == 3:
                    raise e
                print(e)
                child_task = Task.clone(
                    source_task=child_task.id, name=f"{task_name}_lr:{learning_rate:.6f}"
                )

        # Get the loss from the child task
        child_task_results = child_task.get_reported_scalars()
        return child_task_results["loss"]["loss"]["y"][-1]
  
    def get_loss (lr):
        lr = 10 ** lr
        if lr not in loss_per_learning_rate:
            loss_per_learning_rate[lr] = train(lr, template_task_id)
        return loss_per_learning_rate[lr]
    
    for i in range(max_iterations): 
        midpoint = (lower_bound + upper_bound) / 2
        low_loss, up_loss = get_loss(lower_bound), get_loss(upper_bound)
        
        if low_loss < up_loss:
            upper_bound = midpoint
            loss, lr = low_loss, lower_bound
        else:
            lower_bound = midpoint
            loss, lr = up_loss, upper_bound

        if loss < best_loss or i == max_iterations -1 :
            best_loss, best_lr = loss, 10 ** lr
           
        pt_logger.report_scalar("loss", "value", loss, iteration=i)

        print(f"Iteration {i+1}: LR = {lr:.6f}, Loss = {loss:.6f}")
        print(f"Bounds = [{10**lower_bound:.6f}, {10**upper_bound:.6f}]")

        # Check for convergence
        if np.isclose(lower_bound, upper_bound, rtol=1e-5):
            break

    print(f"\nBest learning rate found: {best_lr:.6f} with loss: {best_loss:.6f}")

    parent_task.close()

@hydra.main(config_path="configs", version_base=None)
def main(config):
    config = jax_extra.make_dataclass_from_dict(Config, config)
    project_name, task_name = get_task_details(config) 
    print(f"{project_name=}")
    print(f"{task_name=}")
    template_task_id = Task.get_task(
        project_name=project_name,
        task_name=task_name,
    ).id
 
    lower, upper = np.log10(5e-4), np.log10(5e-2)
    lr_sweep_binary_search(config, lower, upper, template_task_id)
        
if __name__ == "__main__":
    main()
