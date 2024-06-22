class Results:
    __slots__ = ['vqe_times', 'distance_deviations', 'energy_deviations', 'ram_usages', 'cpu_usages',
                 'total_cpu_usages']

    def __init__(self):
        self.vqe_times = []
        self.distance_deviations = []
        self.energy_deviations = []
        self.ram_usages = []
        self.cpu_usages = []
        self.total_cpu_usages = []


class IterationResult:
    __slots__ = ['vqe_time', 'vqe_result', 'vqe_result_energy', 'cpu_usage', 'total_cpu_usage', 'ram_usage']

    def __init__(self, vqe_time, vqe_result, vqe_result_energy, cpu_usage, total_cpu_usage, ram_usage):
        self.vqe_time = vqe_time
        self.vqe_result = vqe_result
        self.vqe_result_energy = vqe_result_energy
        self.cpu_usage = cpu_usage
        self.total_cpu_usage = total_cpu_usage
        self.ram_usage = ram_usage
