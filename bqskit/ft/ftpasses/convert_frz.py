from __future__ import annotations
import numpy as np

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ft.gates.fractional_rz import FractionalRZGate
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.constant.t import TGate
from bqskit.ir.gates.constant.x import XGate
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.ir.point import CircuitPoint

class ConvertFractionalRZPass(BasePass):
    '''
    Pass that converts all FractionalRZGates to a circuit with a QFT
    and GidneyAdders.

    The new circuit will have 3k-1 more qubits than the original circuit.
    Of these, 2k will be permanent data qubits, and k-1 will be ancilla
    qubits that disappear after each adder. I'm not sure how to notate that
    right now.
    '''

    def __init__(self) -> None:
        '''

        Args:
            max_k (int): The maximum value of k for which to 
            generate PKAC circuits.

            add_measurements (bool): Whether or not to add Measurements and
            Resets to the circuit. This is useful for mappers, but not for
            unitary-based subroutines.
        '''
        self.max_k = 3 # TODO: Extend to convert higher k to PK circuits

    def convert_k3_to_t(self, circuit: Circuit) -> None:
        '''
        Convert all FractionalRZGates with k = 3 to T gates. This is just a
        special case of the general conversion, but it is useful to do this
        first since it allows us to use the T gate count as a proxy for how
        well the conversion worked (since T gates are expensive and we want to
        minimize them).
        '''
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, FractionalRZGate) and op.gate.k <= 3:
                # Replace with T gate TODO: fix with actual circuit
                circuit.replace_with_circuit(
                    CircuitPoint(cycle, op.location[0]),
                    FractionalRZGate.get_circuit(op.gate.numerator, op.gate.k),
                    as_circuit_gate=True
                )

        circuit.unfold_all()

    @staticmethod
    def convert_normal_rz(circuit: Circuit, skip_max: int) -> None:
        '''
        Convert all FractionalRZGates with k > skip_max back to 
        normal RZ gates. These then just get synthesized via gridsynth.
        '''
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, FractionalRZGate) and op.gate.k > skip_max:
                # Replace with T gate TODO: fix with actual circuit
                circuit.replace_gate(
                    CircuitPoint(cycle, op.location[0]),
                    RZGate(),
                    op.location,
                    [2 * np.pi * op.gate.numerator/(2 ** op.gate.k)]
                )

    async def run(self, circuit: Circuit, data: PassData) -> None:
        self.convert_k3_to_t(circuit)
        ConvertFractionalRZPass.convert_normal_rz(circuit, self.max_k)