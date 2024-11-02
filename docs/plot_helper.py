from clearml import Task
import matplotlib.pyplot as plt


def get_loss_data(task_ids):
    loss_data = {}
    for task_id in task_ids:
        task = Task.get_task(task_id=task_id)
        scalar_logs = task.get_reported_scalars()

        x_values = scalar_logs["loss"]["loss"]["x"]
        y_values = scalar_logs["loss"]["loss"]["y"]

        task_name = task.name

        loss_data[task_id] = {"name": task_name, "steps": x_values, "loss": y_values}
    return loss_data


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
