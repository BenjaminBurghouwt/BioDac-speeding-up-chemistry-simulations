import time
import os
import psutil
import numpy as np
from tqdm import tqdm

from qiskit_nature.units import DistanceUnit

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator

from scipy.optimize import golden

mapper = JordanWignerMapper()
numpy_solver = NumPyMinimumEigensolver()


def calculate_pc_usage(process):
    # Get the CPU usage of the current process and total CPU usage
    cpu_usage = process.cpu_percent()
    total_cpu_usage = psutil.cpu_percent()

    # Get the memory usage of the current process in GB
    ram_usage = process.memory_info().rss / 1e9

    return cpu_usage, total_cpu_usage, ram_usage


def numpy_eigensolver(distance):
    driver = PySCFDriver(
        atom=f"H 0 0 0; H 0 0 {distance}",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )

    electronic_structure = driver.run()

    calc = GroundStateEigensolver(mapper, numpy_solver)
    res = calc.solve(electronic_structure)

    return res.total_energies[0]


def vqe_calculation(distance):
    driver = PySCFDriver(
        atom=f"H 0 0 0; H 0 0 {distance}",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )

    electronic_structure = driver.run()

    mapper = ParityMapper(num_particles=electronic_structure.num_particles)
    tapered_mapper = electronic_structure.get_tapered_mapper(mapper)

    ansatz = UCCSD(
        electronic_structure.num_spatial_orbitals,
        electronic_structure.num_particles,
        tapered_mapper,
        initial_state=HartreeFock(
            electronic_structure.num_spatial_orbitals,
            electronic_structure.num_particles,
            tapered_mapper,
        ),
    )

    vqe_solver = VQE(Estimator(), ansatz, SLSQP())
    vqe_solver.initial_point = [0.0] * ansatz.num_parameters

    calc = GroundStateEigensolver(tapered_mapper, vqe_solver)
    res = calc.solve(electronic_structure)

    return res.total_energies[0]


def run_vqe_calculations(num_iterations=3):
    process = psutil.Process(os.getpid())

    distance_brack = (0.5, 1)

    # Get the numpy (exact) result
    numpy_result = golden(numpy_eigensolver, brack=distance_brack)
    numpy_result_energy = numpy_eigensolver(numpy_result)

    # Run the VQE calculations multiple times
    vqe_times = []
    distance_deviations = []
    energy_deviations = []
    ram_usages = []
    cpu_usages = []
    total_cpu_usages = []

    smallest_deviation = float('inf')
    closest_distance = None
    closest_energy = None

    for i in tqdm(range(num_iterations)):  # Add a progress bar

        # Run the VQE calculation
        start_time = time.time()
        vqe_result = golden(vqe_calculation, brack=distance_brack)
        end_time = time.time()

        # Calculate the CPU and RAM usage
        cpu_usage, total_cpu_usage, ram_usage = calculate_pc_usage(process)

        if cpu_usage > 0.0:  # Only append if it's a real usage
            cpu_usages.append(cpu_usage)

        total_cpu_usages.append(total_cpu_usage)
        ram_usages.append(ram_usage)

        # Append the results
        vqe_times.append(end_time - start_time)

        vqe_result_energy = vqe_calculation(vqe_result)

        deviation_distance = abs(vqe_result - numpy_result)
        distance_deviations.append(deviation_distance)

        deviation_energy = abs(vqe_result_energy - numpy_result_energy)
        energy_deviations.append(deviation_energy)

        if deviation_distance < smallest_deviation:
            smallest_deviation = deviation_distance
            closest_distance = vqe_result
            closest_energy = vqe_result_energy

    largest_deviation_distance = max(distance_deviations)
    largest_deviation_energy = max(energy_deviations)

    average_time = sum(vqe_times) / len(vqe_times)
    average_ram_usage = sum(ram_usages) / len(ram_usages)
    average_cpu_usage = sum(cpu_usages) / len(cpu_usages)
    average_total_cpu_usage = sum(total_cpu_usages) / len(total_cpu_usages)

    print(f"Largest deviation from exact result (distance): {largest_deviation_distance}")
    print(f"Largest deviation from exact result (energy): {largest_deviation_energy}")
    print(f"Average computation time: {average_time} seconds")

    print("\nBest VQE result:")
    print("Inter atomic distance: ", closest_distance, "Angstrom - Exact: ", numpy_result, "Angstrom")
    print("Energy: ", closest_energy, "Hartree - Exact: ", numpy_result_energy, "Hartree")

    print("\nPerformance metrics:")
    print(f"Average RAM usage: {average_ram_usage} GB")
    print(f"Average python CPU usage: {average_cpu_usage} %")
    print(f"Average total CPU usage: {average_total_cpu_usage} %")

    print("\n --- ")
    print("Due to printing a progress bar and keeping track of the CPU usage, the computation time is higher than it "
          "would be without these features.")


def main():
    run_vqe_calculations(num_iterations=50)


if __name__ == "__main__":
    main()
