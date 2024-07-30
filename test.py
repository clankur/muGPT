# %%
from clearml import Task

# %%
task: Task = Task.get_task("020a9883f8584b1ab9e53613db3c7edd")

# %%
task.set_system_tags([])
# %%
task_list = Task.get_tasks(project_name="c4_a100x8x4_37m/mup", allow_archived=True)
[(task.name, task.id) for task in task_list]

# %%
lr = 0.00197411
lr = round(lr, 5)
print(f"{lr:.7f}")
task = Task.get_task(task_name=f"37m_mup_base_lr:{lr:.7f}")

# %%
tasks = Task.get_tasks(
    system_tags=["__hidden__"],
    filters={
        "status": ["completed", "published", "failed", "stopped"],
        "runtime.progress": {"$gt": 0},
    },
)
for task in tasks:
    print(task.name)
    task.set_system_tags([])

# %%
# Get all tasks
tasks = Task.get_tasks(task_filter={"parent": '07217373eb714710bda2b1acb1573a06'})
tasks
# Filter tasks with system hidden tag and >0 iterations
hidden_tasks = [
    task
    for task in tasks
    if "__hidden__" in task.get_system_tags() and task.get_last_iteration() > 0
]
hidden_tasks

# %%
# Get all tasks
all_tasks = Task.get_tasks(
    project_name=,
    task_filter={'status': ['completed', 'published', 'failed', 'stopped']
})

# Filter tasks with system hidden tag and >0 iterations
hidden_tasks = [
    task for task in all_tasks
    if '__hidden__' in task.get_system_tags() and task.get_last_iteration() > 0
]

# %%
[task.name for task in all_tasks if "base" in task.name ]
# %%
Task.get_project_id("c4_a100x8x4_37m/mup", search_hidden=True)
# %%
Task.__get_tasks(project_name="c4_a100x8x4_37m/mup")
# %%
