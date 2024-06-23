import numpy as np
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import Aer
from qiskit_ibm_runtime import Session, EstimatorV2
from scipy.optimize import minimize


cost_history = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": [],
}

simulated_backend = Aer.get_backend("qasm_simulator")


def cost_func(params, ansatz, hamiltonian, estimator, nuclear_repulsion_energy):
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0] + nuclear_repulsion_energy

    cost_history["iters"] += 1
    cost_history["prev_vector"] = params
    cost_history["cost_history"].append(energy)

    return energy


def vqe_calculation(distance, hydrogen_system, backend=simulated_backend):
    # reset cost history
    global cost_history
    cost_history = {
        "prev_vector": None,
        "iters": 0,
        "cost_history": [],
    }

    hydrogen_system.update_for_distance(distance)

    ansatz = hydrogen_system.ansatz
    hamiltonian = hydrogen_system.electronic_structure.hamiltonian.second_q_op()
    mapper = hydrogen_system.mapper

    nuclear_repulsion_energy = hydrogen_system.electronic_structure.nuclear_repulsion_energy

    qubit_operator = mapper.map(hamiltonian)

    hamiltonian_list = []

    for pauli, coefficient in sorted(qubit_operator.label_iter()):
        hamiltonian_list.append((pauli, coefficient.real))

    hamiltonian = SparsePauliOp.from_list(hamiltonian_list)

    num_params = ansatz.num_parameters

    target = backend.target
    pm = generate_preset_pass_manager(target=target, optimization_level=3)

    ansatz_isa = pm.run(ansatz)

    # quantum inspire backend does not have a target, so we can't use the pass manager
    # ansatz_isa = transpile(ansatz, backend=backend)

    hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)

    initial_point = np.zeros(num_params)

    with Session(backend=backend) as session:
        estimator = EstimatorV2(mode=session)
        # estimator.options.default_shots = 10000

        result = minimize(
            cost_func,
            initial_point,
            args=(ansatz_isa, hamiltonian_isa, estimator, nuclear_repulsion_energy),
            method="cobyla"
        )

    return result, cost_history
