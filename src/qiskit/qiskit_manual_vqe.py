import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator, Session
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.units import DistanceUnit
from scipy.optimize import minimize

cost_history_dict = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": [],
}


def cost_func(params, ansatz, hamiltonian, estimator, nuclear_repulsion_energy):
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0] + nuclear_repulsion_energy

    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(energy)

    return energy


def manual_vqe(distance, backend):
    driver = PySCFDriver(
        atom=f"H 0 0 0; H 0 0 {distance}",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )

    electronic_structure = driver.run()

    nuclear_repulsion_energy = electronic_structure.nuclear_repulsion_energy

    # mapper = JordanWignerMapper()
    mapper = ParityMapper(num_particles=electronic_structure.num_particles)
    tapered_mapper = electronic_structure.get_tapered_mapper(mapper)

    # qubit_op = mapper.map(electronic_structure.hamiltonian.second_q_op())
    # reduced_op = mapper.map(electronic_structure.hamiltonian.second_q_op())
    tapered_op = tapered_mapper.map(electronic_structure.hamiltonian.second_q_op())

    hamiltonian_list = []

    for pauli, coeff in sorted(tapered_op.label_iter()):
        hamiltonian_list.append((pauli, coeff.real))

    hamiltonian = SparsePauliOp.from_list(hamiltonian_list)

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

    num_params = ansatz.num_parameters

    target = backend.target
    pm = generate_preset_pass_manager(target=target, optimization_level=3)

    ansatz_isa = pm.run(ansatz)

    hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)

    # x0 = 2 * np.pi * np.random.random(num_params)
    x0 = np.zeros(num_params)

    with Session(backend=backend) as session:
        estimator = Estimator(mode=session)
        estimator.options.default_shots = 10000

        res = minimize(
            cost_func,
            x0,
            args=(ansatz_isa, hamiltonian_isa, estimator, nuclear_repulsion_energy),
            method="cobyla",
        )

    return res, cost_history_dict
