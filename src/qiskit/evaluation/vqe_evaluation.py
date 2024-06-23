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

from src.qiskit.evaluation import manual_vqe, pc_usage
from src.qiskit.evaluation.evaluation_summary import print_summary
from src.qiskit.evaluation.results import IterationResult, Results
from src.qiskit.evaluation.hydrogen_system.hydrogen_system import HydrogenSystem

hydrogen_num_particles = (1, 1)
hydrogen_distance_bracket = (0.5, 1)

numpy_result = None
numpy_result_energy = None

jordan_wigner_mapper = JordanWignerMapper()
parity_mapper = ParityMapper(num_particles=hydrogen_num_particles)
numpy_solver = NumPyMinimumEigensolver()

hydrogen_system = HydrogenSystem(0.735, parity_mapper, use_tapered_mapper=True)


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


def manual_vqe_calculation(distance):
    hydrogen_system.update_for_distance(distance)

    result, cost_history = manual_vqe.vqe_calculation(distance, hydrogen_system, manual_vqe.simulated_backend)

    return result["fun"]


def run_vqe_iteration(process, calculation_function):
    start_time = time.time()
    vqe_result = golden(calculation_function, brack=hydrogen_distance_bracket)
    end_time = time.time()

    cpu_usage, total_cpu_usage, ram_usage = pc_usage.calculate(process)
    vqe_result_energy = calculation_function(vqe_result)

    return IterationResult(
        vqe_time=end_time - start_time,
        vqe_result=vqe_result,
        vqe_result_energy=vqe_result_energy,
        cpu_usage=cpu_usage,
        total_cpu_usage=total_cpu_usage,
        ram_usage=ram_usage
    )


def run_iterations(num_iterations, calculation_function):
    process = psutil.Process(os.getpid())

    results = Results()

    smallest_deviation = float('inf')
    closest_result = {"distance": None, "energy": None}

    for i in tqdm(range(num_iterations)):
        iteration_result = run_vqe_iteration(process, calculation_function)

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

    return results, closest_result


def run_vqe_evaluation(num_iterations=3, evaluate_qiskit_nature=True, evaluate_manual_vqe=False):
    if not evaluate_qiskit_nature and not evaluate_manual_vqe:
        print("Both evaluation methods are disabled. Exiting...")
        return

    global numpy_result, numpy_result_energy

    # Get the numpy (exact) result
    numpy_result = golden(numpy_eigensolver, brack=hydrogen_distance_bracket)
    numpy_result_energy = numpy_eigensolver(numpy_result)

    if evaluate_qiskit_nature:
        results_vqe, closest_result_vqe = run_iterations(num_iterations, vqe_calculation)

        print("VQE results:")
        print_summary(results_vqe, numpy_result, numpy_result_energy, closest_result_vqe)

    if evaluate_manual_vqe:
        results_manual_vqe, closest_result_manual_vqe = run_iterations(num_iterations, manual_vqe_calculation)

        print("Manual VQE results:")
        print_summary(results_manual_vqe, numpy_result, numpy_result_energy, closest_result_manual_vqe)


def main():
    run_vqe_evaluation(num_iterations=3, evaluate_qiskit_nature=True, evaluate_manual_vqe=True)


if __name__ == "__main__":
    main()
