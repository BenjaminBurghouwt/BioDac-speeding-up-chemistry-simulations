import os
from time import sleep

from dotenv import load_dotenv
from qiskit.visualization import plot_histogram

from quantuminspire.credentials import enable_account
from quantuminspire.credentials import get_authentication
from quantuminspire.qiskit import QI

from qiskit import transpile, QuantumCircuit
from qiskit_nature.units import DistanceUnit
from qiskit_aer import Aer, AerSimulator

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator

load_dotenv()
api_token = os.getenv('QI_API_TOKEN')

project_name = 'Orbitals - Speeding up Chemistry Simulations'

enable_account(api_token)
QI.set_authentication()

mapper = JordanWignerMapper()


def vqe_backend_calculations(distance):
    driver = PySCFDriver(
        atom=f"H 0 0 0; H 0 0 {distance}",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )

    electronic_structure = driver.run()

    ansatz = UCCSD(
        electronic_structure.num_spatial_orbitals,
        electronic_structure.num_particles,
        mapper,
        initial_state=HartreeFock(
            electronic_structure.num_spatial_orbitals,
            electronic_structure.num_particles,
            mapper,
        ),
    )


if __name__ == "__main__":
    print(vqe_backend_calculations(0.7348651644548676))

# --------------------------------------------------------


mapper = JordanWignerMapper()


def vqe_calculation(distance):
    driver = PySCFDriver(
        atom=f"H 0 0 0; H 0 0 {distance}",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )

    electronic_structure = driver.run()

    ansatz = UCCSD(
        electronic_structure.num_spatial_orbitals,
        electronic_structure.num_particles,
        mapper,
        initial_state=HartreeFock(
            electronic_structure.num_spatial_orbitals,
            electronic_structure.num_particles,
            mapper,
        ),
    )

    vqe_solver = VQE(Estimator(), ansatz, SLSQP())
    vqe_solver.initial_point = [0.0] * ansatz.num_parameters

    vqe_solver.get_qubit_operators()

    # calc = GroundStateEigensolver(mapper, vqe_solver)

    # circuit = transpile(vqe_solver.ansatz, backend=backend)

    return vqe_solver.ansatz

# --------------------------------------------------------


# if __name__ == "__main__":
#     # authentication = get_authentication()
#     # QI.set_authentication(authentication, project_name=project_name)
#     # qi_backend = QI.get_backend('QX single-node simulator')
#
#     ansatz = vqe_calculation(0.7348651644548676)
#     # ansatz.measure_all()
#
#     backend = Aer.get_backend('statevector_simulator')
#     simulator = AerSimulator()
#
#     circuit = transpile(ansatz, backend=simulator)
#
#     print(circuit.layout)
#
#     # circuit.draw('mpl').show()
#
#     # job = backend.run(circuit, shots=1024)
#
#     # print(job.status())
#
#     # time = 0
#     # wait = 3
#     # while time < 300:
#     #     if job.done():
#     #         counts = job.result().get_counts()
#     #         print(counts)
#     #         break
#     #
#     #     sleep(wait)
#     #     time += wait
