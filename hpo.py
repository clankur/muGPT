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


def get_base_template_id(config: Config):
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
    return Task.get_task(
        project_name=f"{config_name}/{git_branch_name}",
        task_name=task_name,
    ).id



def lr_sweep_binary_search(
    lower_bound, upper_bound, template_task_id, max_iterations=5, queue_name="default"
):
    parent_task = Task.init(project_name="LR Sweep", task_name="Binary Search LR Sweep")
    pt_logger = parent_task.get_logger()
    best_lr = None
    best_loss = float("inf")

    loss_per_learning_rate = {} 
    
    def train (learning_rate, template_task_id):
        # Clone the template task and override the learning rate
        child_task = Task.clone(
            source_task=template_task_id, name=f"LR Sweep - LR {learning_rate:.6f}"
        )
        child_task.set_parameter("Hydra/training.learning_rate", learning_rate ) 
        print(f"training model with lr: {learning_rate}")
        # Enqueue the child task for execution
        Task.enqueue(child_task.id, queue_name=queue_name)

        # Wait for the child task to complete
        child_task.wait_for_status()

        # Get the results from the child task
        child_task_results = child_task.get_reported_scalars()
        # print(f"{child_task_results=}")
        # print(f"{child_task_results['loss']=}")
        return child_task_results["loss"]["loss"]["y"][-1]
   
    for i in range(max_iterations): 
        lr_samples = 10 ** np.random.uniform(np.log10(lower_bound), np.log10(upper_bound), 5)

        for lr in lr_samples:
            loss = train(lr, template_task_id)
            loss_per_learning_rate[lr] = loss
        
            # Log the results in the parent task
            pt_logger.report_scalar(
                "learning_rate", "value", lr, iteration=i
            )
            if loss < best_loss:
                best_lr = lr 
                best_loss = loss
            
        pt_logger.report_scalar("loss", "value", loss, iteration=i)

        print(f"Iteration {i+1}: LR = {best_lr:.6f}, Loss = {best_loss:.6f}")
        print(f"Bounds = [{lower_bound:.6f}, {upper_bound:.6f}]")
        # Update best learning rate if necessary
        lower_bound = np.power(10, np.log10(best_lr) - .5)
        upper_bound = np.power(10, np.log10(best_lr) + .5)

        # Check for convergence
        if np.isclose(lower_bound, upper_bound, rtol=1e-5):
            break

    print(f"\nBest learning rate found: {best_lr:.6f} with loss: {best_loss:.6f}")
    pt_logger.report_scalar("best_learning_rate", "value", best_lr)
    pt_logger.get_logger().report_scalar("best_loss", "value", best_loss)

    parent_task.close()


@hydra.main(config_path="configs", version_base=None)
def main(config):
    config = jax_extra.make_dataclass_from_dict(Config, config)
    template_task_id = get_base_template_id(config) 
    lr_sweep_binary_search(1e-5, 1.0, template_task_id, queue_name=config.training.queue)
    
if __name__ == "__main__":
    main()
