from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ft.gadgets.qft import QFTGadget
from bqskit.ft.gates.fractional_rz import FractionalRZGate
from bqskit.ft.gates.gidney_adder import GidneyAdder
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.t import TGate
from bqskit.ir.gates.constant.x import XGate
from bqskit.ir.gates.measure import MeasurementPlaceholder
from bqskit.ir.point import CircuitPoint


class ConvertToPKAC(BasePass):
    '''
    Pass that converts all FractionalRZGates to a circuit with a QFT
    and GidneyAdders.

    The new circuit will have 3k-1 more qubits than the original circuit.
    Of these, 2k will be permanent data qubits, and k-1 will be ancilla
    qubits that disappear after each adder. I'm not sure how to notate that
    right now.
    '''

    def __init__(self, ks: list[int] = [3], add_measurements: bool = True) -> None:
        '''

        Args:
            ks (list[int]): Will be the register sizes for the Adders used to perform
            RZ gates. Can perform rotation multiples of 2*pi/(2 ** k). If k=3,
            can perform T gates.

            add_measurements (bool): Whether or not to add Measurements and
            Resets to the circuit. This is useful for mappers, but not for
            unitary-based subroutines.
        '''
        self.ks = ks
        self.add_measurements = add_measurements

    def calculate_cnot_circuit(self, numerator: int, k: int) -> Circuit:
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
        circ = Circuit(k + 1)
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
            target_ind = k - i
            if bit == 1:
                circ.append_gate(CNOTGate(), [0, target_ind])
            elif bit == -1:
                circ.append_gate(XGate(), [target_ind])
                circ.append_gate(CNOTGate(), [0, target_ind])

        return circ

    def convert_k3_to_t(self, circuit: Circuit) -> None:
        '''
        Convert all FractionalRZGates with k = 3 to T gates. This is just a
        special case of the general conversion, but it is useful to do this
        first since it allows us to use the T gate count as a proxy for how
        well the conversion worked (since T gates are expensive and we want to
        minimize them).
        '''
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, FractionalRZGate) and op.gate.k == 3:
                # Replace with T gate TODO: fix with actual circuit
                circuit.replace_gate(
                    CircuitPoint(cycle, op.location[0]),
                    TGate(),
                    op.location
                )


    async def run(self, circuit: Circuit, data: PassData) -> None:
        # New circuit size is num_qudits + k (input a) + sum(qft_state_sizes) input bs + k - 1 ancillas

        # We need to first convert all FractionalRZGates with k = 3
        # to circuits with T gates

        if 3 in self.ks:
            self.convert_k3_to_t(circuit)
            self.ks.remove(3)
            if len(self.ks) == 0:
                return
        
        max_k = max(self.ks)
        circuit_size = circuit.num_qudits + max_k + sum(self.ks) + max_k - 1
        new_circ = Circuit(circuit_size)

        n = circuit.num_qudits

        input_a_qubits = list(range(n, n + max_k))
        all_input_q_qubits = []
        start = n + max_k
        for size in self.ks:
            end = start + size
            all_input_q_qubits.append(list(range(start, end)))
            start = end
        ancilla_qubits = list(range(start, circuit_size))

        # Initialize input Bs in QFT state
        for size, input_b_qubits in zip(self.ks, all_input_q_qubits):
            new_circ.append_gate(XGate(), [input_b_qubits[-1]])
            new_circ.append_circuit(QFTGadget.generate(size), input_b_qubits)

        # Initialize ancilla qubits with measurements
        if self.add_measurements:
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

                # Get correct k to use
                qft_ind = self.ks.index(op.gate.k)
                input_b_qubits = all_input_q_qubits[qft_ind]
                new_k = self.ks[qft_ind]

                # Add a CNOT to LSB on input A
                num = op.gate.numerator % (2 ** (new_k - 1))
                cnots = self.calculate_cnot_circuit(num, new_k)

                new_circ.append_circuit(cnots, [q] + input_a_qubits[:new_k])

                input_b = all_input_q_qubits[qft_ind]

                # Apply adder on A, B, and ancilla
                new_circ.append_gate(
                    GidneyAdder(new_k, add_reset=self.add_measurements),
                    (
                        (input_a_qubits[:new_k] + input_b + 
                         ancilla_qubits[:new_k - 1])
                    ),
                )

                new_circ.append_circuit(cnots, [q] + input_a_qubits[:new_k])
            else:
                new_circ.append(op)

        circuit.become(new_circ)
