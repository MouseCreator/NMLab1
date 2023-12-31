import re


def parse_task_file(filename):
    tasks = []
    current_task = {}

    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.endswith("}"):
            tasks.append(current_task)
            current_task = {}
            continue
        if line.endswith("{"):
            line = line.replace("{", "").strip()
            current_task["name"] = line
            continue
        key, value = re.split(r'\s*:\s*', line)
        current_task[key] = value

    if current_task:
        tasks.append(current_task)

    return tasks


