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


def get_metrics(task_ids, truncate=False, get_synthetic=False):
    metrics = {}
    min_n_values = float("inf")
    for task_id in task_ids:
        task = Task.get_task(task_id=task_id)
        scalar_logs = task.get_reported_scalars(max_samples=5000)
        task_name = task.name

        metrics[task_id] = {
            "name": task_name,
            "Loss": scalar_logs["loss"]["loss"],
        }

        if get_synthetic:
            metrics[task_id].update(
                {
                    "Avg Total Confidence": scalar_logs["Avg Total Confidence"][
                        "Average total confidence"
                    ],
                    "Max char confidence": scalar_logs["Character confidence"][
                        "Max char confidence"
                    ],
                    "Average char confidence": scalar_logs["Character confidence"][
                        "Average char confidence"
                    ],
                }
            )

        min_n_values = min(max(metrics[task_id]["Loss"]["x"]), min_n_values)

    if truncate:
        for task_id, data in metrics.items():
            index = next(
                (i for i, x in enumerate(data["Loss"]["x"]) if x >= min_n_values), None
            )
            for key in data:
                if isinstance(data[key], dict):  # Apply truncation to dict-like metrics
                    data[key]["x"] = data[key]["x"][:index]
                    data[key]["y"] = data[key]["y"][:index]

    return metrics


def calculate_ema(data, smoothing=0.97):
    ema = [data[0]]
    for value in data[1:]:
        ema.append(ema[-1] * smoothing + value * (1 - smoothing))
    return ema


def plot_data(
    metrics, metric_key, plot_last: int = 1000, x_series="x", y_series="y", title=None
):
    """
    Plots the specified metric across all task_ids.

    Args:
        metrics (dict): Metrics dictionary returned from `get_metrics`.
        metric_key (str): Key of the metric to plot (e.g., "Loss", "Avg Total Confidence").
        plot_last (int): Number of last steps to plot. Defaults to 1000.
        x_series (str): Key for the x-axis values. Defaults to "x".
        y_series (str): Key for the y-axis values. Defaults to "y".
        title (str): Title of the plot. Defaults to `metric_key` if not provided.
    """
    plt.figure(figsize=(10, 6))
    for task_id, data in metrics.items():
        if metric_key not in data:
            continue

        steps = data[metric_key][x_series][-plot_last:]
        values = data[metric_key][y_series][-plot_last:]

        # Calculate EMA for smoother visualization
        values_ema = calculate_ema(values, smoothing=0.97)

        (line,) = plt.plot(steps, values, alpha=0.25, label=f"{data['name']}")
        color = line.get_color()
        plt.plot(steps, values_ema, color=color, linestyle="--")

    plt.title(title or metric_key)
    plt.xlabel("Steps")
    plt.ylabel(metric_key)
    plt.legend(title="Experiments")
    plt.minorticks_on()
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.show()


def plot_all_metrics(metrics, plot_last=1000, x_series="x", y_series="y"):
    """
    Plots all metrics available in the metrics dictionary by reusing `plot_data`.

    Args:
        metrics (dict): Metrics dictionary returned from `get_metrics`.
        plot_last (int): Number of last steps to plot. Defaults to 1000.
        x_series (str): Key for the x-axis values. Defaults to "x".
        y_series (str): Key for the y-axis values. Defaults to "y".
    """
    if not metrics:
        print("No metrics data to plot.")
        return

    # Extract all unique metric keys from the first task's metrics
    task_sample = next(iter(metrics.values()))
    metric_keys = [key for key in task_sample if isinstance(task_sample[key], dict)]

    for metric_key in metric_keys:
        print(f"Plotting {metric_key}...")
        plot_data(
            metrics,
            metric_key=metric_key,
            plot_last=plot_last,
            x_series=x_series,
            y_series=y_series,
            title=f"{metric_key}",
        )
