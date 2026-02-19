from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ft.gadgets.qft import QFTGadget
from bqskit.ft.gates.fractional_rz import FractionalRZGate
from bqskit.ft.gates.gidney_adder import GidneyAdder
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.x import XGate
from bqskit.ir.gates.measure import MeasurementPlaceholder


class ConvertToPKAC(BasePass):
    '''
    Pass that converts all FractionalRZGates to a circuit with a QFT
    and GidneyAdders.

    The new circuit will have 3k-1 more qubits than the original circuit.
    Of these, 2k will be permanent data qubits, and k-1 will be ancilla
    qubits that disappear after each adder. I'm not sure how to notate that
    right now.
    '''

    def __init__(self, k: int = 3) -> None:
        self.k = k

    def calculate_cnot_circuit(self, numerator: int) -> Circuit:
        '''
        Calculate the circuit of CNOTs needed to apply the appropriate RZ
        rotation. Starting with LSB of target, applying a CNOT applies an
        RZ of 2*pi / 2^k on the control qubit. LSB + 1 applies an RZ of
        2*pi / 2^(k-1), and so forth. If you want to apply RZ(-2*pi / 2^k),
        you can apply an X on the LSB and then a CNOT. We make the circuit
        to minimize the number of CNOTs.

        Args:
            numerator (int): The numerator of the angle to apply, where the
                denominator is 2^k. For example, if k = 3 and you want to apply
                RZ(pi/4), the numerator would be 1, since pi/4 = 2*pi / 2^3.

        '''
        circ = Circuit(self.k + 1)
        # Let 0 be the control and 1...k be the target register for the adder

        # Run NAF algorithm to find the optimal bitstring
        bitstring = []
        n = numerator
        while n > 0:
            if n % 2 == 0:
                bitstring.append(0)
                n = n // 2
            else:
                r = 2 - (n % 4)
                bitstring.append(2 - (n % 4))
                n = (n - r) // 2

        # Now bitstring[i] tells us whether we need to apply a CNOT with target
        for i, bit in enumerate(bitstring):
            # LSB is the last bit of register
            target_ind = self.k - i
            if bit == 1:
                circ.append_gate(CNOTGate(), [0, target_ind])
            elif bit == -1:
                circ.append_gate(XGate(), [target_ind])
                circ.append_gate(CNOTGate(), [0, target_ind])

        return circ

    async def run(self, circuit: Circuit, data: PassData) -> None:
        new_circ = Circuit(circuit.num_qudits + 3 * self.k - 1)

        n = circuit.num_qudits

        input_a_qubits = list(range(n, n + self.k))
        input_b_qubits = list(range(n + self.k, n + 2 * self.k))
        ancilla_qubits = list(range(n + 2 * self.k, n + 3 * self.k - 1))

        # Initialize input B in QFT state
        new_circ.append_gate(XGate(), [input_b_qubits[-1]])
        new_circ.append_circuit(QFTGadget.generate(self.k), input_b_qubits)

        # Initialize ancilla qubits with measurements
        for q in ancilla_qubits:
            new_circ.append_gate(
                MeasurementPlaceholder(
                    [('a', 1)], {q: ('a', 0)},
                ), [q],
            )

        for op in circuit.operations():
            if isinstance(op.gate, FractionalRZGate):
                # Replace with adder circuit
                q = op.location[0]
                # Add a CNOT to LSB on input A
                cnots = self.calculate_cnot_circuit(op.gate.numerator)
                new_circ.append_circuit(cnots, [q] + input_a_qubits)

                # Apply adder on A, B, and ancilla
                new_circ.append_gate(
                    GidneyAdder(self.k),
                    (
                        input_a_qubits + input_b_qubits
                        + ancilla_qubits
                    ),
                )

                new_circ.append_circuit(cnots, [q] + input_a_qubits)
            else:
                new_circ.append(op)

        circuit.become(new_circ)
