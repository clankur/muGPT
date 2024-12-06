from clearml import Task
import matplotlib.pyplot as plt
import yaml


def get_configs(task_ids):
    configs = {}
    for task_id in task_ids:
        task = Task.get_task(task_id=task_id)
        task_name = task.name
        task_settings = task.export_task()

        task_config = yaml.safe_load(
            task_settings["configuration"]["OmegaConf"]["value"]
        )
        task_config = {
            "training": task_config.get("training"),
            "model": task_config.get("model"),
        }
        task_config = yaml.dump(task_config, default_flow_style=False)

        configs[task_id] = {
            "name": task_name,
            "config": task_config,
        }
    return configs


def get_loss_data(task_ids, truncate=False):
    loss_data = {}
    min_n_values = float("inf")
    for task_id in task_ids:
        task = Task.get_task(task_id=task_id)
        scalar_logs = task.get_reported_scalars(max_samples=5000)

        x_values = scalar_logs["loss"]["loss"]["x"]
        y_values = scalar_logs["loss"]["loss"]["y"]

        task_name = task.name

        loss_data[task_id] = {"name": task_name, "steps": x_values, "loss": y_values}
        min_n_values = min(max(x_values), min_n_values)
    if truncate:
        for task_id, data in loss_data.items():
            index = next(
                (i for i, x in enumerate(data["steps"]) if x >= min_n_values), None
            )
            data["steps"] = data["steps"][:index]
            data["loss"] = data["loss"][:index]
    return loss_data


def get_synthetic_metrics(task_ids):
    synthetic_data = {}
    for task_id in task_ids:
        task = Task.get_task(task_id=task_id)
        scalar_logs = task.get_reported_scalars()
        print(scalar_logs)


def calculate_ema(data, smoothing=0.97):
    ema = [data[0]]
    for value in data[1:]:
        ema.append(ema[-1] * smoothing + value * (1 - smoothing))
    return ema


def plot_loss_data(loss_data, plot_last: int = 1000):
    plt.figure(figsize=(10, 6))
    for _, data in loss_data.items():
        steps = data["steps"][-plot_last:]
        loss = data["loss"][-plot_last:]

        loss_ema = calculate_ema(loss, smoothing=0.97)

        (line,) = plt.plot(steps, loss, alpha=0.5, label=f"{data['name']}")
        color = line.get_color()
        plt.plot(steps, loss_ema, color=color)

    plt.title("Loss Over Steps for Each Task")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend(title="Experiments")
    plt.minorticks_on()
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.show()
