import psutil


def calculate(process):
    # Get the CPU usage of the current process and total CPU usage
    cpu_usage = process.cpu_percent()
    total_cpu_usage = psutil.cpu_percent()

    # Get the memory usage of the current process in GB
    ram_usage = process.memory_info().rss / 1e9

    return cpu_usage, total_cpu_usage, ram_usage