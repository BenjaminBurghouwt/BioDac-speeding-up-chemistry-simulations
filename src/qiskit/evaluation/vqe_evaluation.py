import time
import os
import psutil
from tqdm import tqdm

from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator

from scipy.optimize import golden

import pc_usage
from evaluation_summary import print_summary
from hydrogen_system.hydrogen_system import HydrogenSystem
from results import IterationResult, Results

hydrogen_num_particles = (1, 1)
hydrogen_distance_bracket = (0.5, 1)

jordan_wigner_mapper = JordanWignerMapper()
parity_mapper = ParityMapper(num_particles=hydrogen_num_particles)
numpy_solver = NumPyMinimumEigensolver()

hydrogen_system = HydrogenSystem(0.1, jordan_wigner_mapper, use_tapered_mapper=False)


def numpy_eigensolver(distance):
    hydrogen_system.update_for_distance(distance)

    calc = GroundStateEigensolver(hydrogen_system.mapper, numpy_solver)
    res = calc.solve(hydrogen_system.electronic_structure)

    return res.total_energies[0]


def vqe_calculation(distance):
    hydrogen_system.update_for_distance(distance)

    ansatz = hydrogen_system.ansatz

    vqe_solver = VQE(Estimator(), ansatz, SLSQP())
    vqe_solver.initial_point = [0.0] * ansatz.num_parameters

    calc = GroundStateEigensolver(hydrogen_system.mapper, vqe_solver)
    res = calc.solve(hydrogen_system.electronic_structure)

    return res.total_energies[0]


def run_vqe_iteration(process):
    start_time = time.time()
    vqe_result = golden(vqe_calculation, brack=hydrogen_distance_bracket)
    end_time = time.time()

    cpu_usage, total_cpu_usage, ram_usage = pc_usage.calculate(process)
    vqe_result_energy = vqe_calculation(vqe_result)

    return IterationResult(
        vqe_time=end_time - start_time,
        vqe_result=vqe_result,
        vqe_result_energy=vqe_result_energy,
        cpu_usage=cpu_usage,
        total_cpu_usage=total_cpu_usage,
        ram_usage=ram_usage
    )


def run_vqe_evaluation(num_iterations=3):
    process = psutil.Process(os.getpid())

    # Get the numpy (exact) result
    numpy_result = golden(numpy_eigensolver, brack=hydrogen_distance_bracket)
    numpy_result_energy = numpy_eigensolver(numpy_result)

    results = Results()

    smallest_deviation = float('inf')
    closest_result = {"distance": None, "energy": None}

    for i in tqdm(range(num_iterations)):
        iteration_result = run_vqe_iteration(process)

        if iteration_result.cpu_usage > 0.0:
            results.cpu_usages.append(iteration_result.cpu_usage)

        results.total_cpu_usages.append(iteration_result.total_cpu_usage)
        results.ram_usages.append(iteration_result.ram_usage)
        results.vqe_times.append(iteration_result.vqe_time)

        deviation_distance = abs(iteration_result.vqe_result - numpy_result)
        deviation_energy = abs(iteration_result.vqe_result_energy - numpy_result_energy)

        results.distance_deviations.append(deviation_distance)
        results.energy_deviations.append(deviation_energy)

        if deviation_distance < smallest_deviation:
            smallest_deviation = deviation_distance
            closest_result["distance"] = iteration_result.vqe_result
            closest_result["energy"] = iteration_result.vqe_result_energy

    print_summary(results, numpy_result, numpy_result_energy, closest_result)

def main():
    run_vqe_evaluation()


if __name__ == "__main__":
    main()
