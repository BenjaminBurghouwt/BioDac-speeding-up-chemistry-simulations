from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit


class HydrogenSystem:
    def __init__(self, distance, mapper, use_tapered_mapper=False):
        self.distance = distance
        self.driver = self.create_driver()
        self.electronic_structure = self.driver.run()
        self.mapper = mapper

        if use_tapered_mapper:
            self.mapper = self.electronic_structure.get_tapered_mapper(mapper)

        self.ansatz = self.create_ansatz()

    def create_driver(self):
        return PySCFDriver(
            atom=f"H 0 0 0; H 0 0 {self.distance}",
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        )

    def create_ansatz(self):
        return UCCSD(
            self.electronic_structure.num_spatial_orbitals,
            self.electronic_structure.num_particles,
            self.mapper,
            initial_state=HartreeFock(
                self.electronic_structure.num_spatial_orbitals,
                self.electronic_structure.num_particles,
                self.mapper,
            ),
        )

    def update_for_distance(self, distance):
        self.distance = distance
        self.driver = self.create_driver()
        self.electronic_structure = self.driver.run()
        self.ansatz = self.create_ansatz()
