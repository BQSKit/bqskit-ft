from __future__ import annotations
import numpy as np

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ft.gadgets.qft import QFTGadget
from bqskit.ft.gates.fractional_rz import FractionalRZGate
from bqskit.ft.gates.gidney_adder import GidneyAdder
from bqskit.ft.gates.logical_and import LogicalAndDgGate, LogicalAndGate
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.constant.t import TGate
from bqskit.ir.gates.constant.tdg import TdgGate
from bqskit.ir.gates.constant.tdg import TdgGate
from bqskit.ir.gates.constant.x import XGate
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.ir.gates.measure import MidCircuitMeasurement
from bqskit.ir.point import CircuitPoint


def sqrt_circuit() -> Circuit:
    '''
    Circuit to apply a sqrt(T) gate using a cataylzed state. This is used in the
    case where we want to perform RZ rotations with k = 4, but we don't want
    to use the full PKAC construction since it is expensive and we only need
    one rotation.

    The circuit takes in the target qubit, an second qubit to store a second
    sqrt(T) state, and a third qubit which is the catalyzed sqrt(T) state.
    The last qubit is an ancilla to use in the logical AND.

    TODO: Maybe we can use a 3-qubit circuit instead (no qubit 1)
    '''
    circ = Circuit(4)


    # Initialize ancilla in T state
    circ.append_gate(HGate(), [3])
    circ.append_gate(TGate(), [3])

    # Run circuit
    circ.append_gate(XGate(), [2])
    circ.append_gate(CNOTGate(), [0, 1])
    circ.append_gate(CNOTGate(), [0, 2])
    circ.append_gate(LogicalAndGate(), [1, 2, 3])


    circ.append_gate(CNOTGate(), [0, 3])
    circ.append_gate(TGate(), [3])
    circ.append_gate(CNOTGate(), [0, 3])

    circ.append_gate(LogicalAndDgGate(), [1, 2, 3])

    circ.append_gate(CNOTGate(), [1, 2])
    circ.append_gate(CNOTGate(), [0, 1])
    circ.append_gate(CNOTGate(), [0, 2])
    circ.append_gate(XGate(), [2])
    return circ

class ConvertToPKAC(BasePass):
    '''
    Pass that converts all FractionalRZGates to a circuit with a QFT
    and GidneyAdders.

    The new circuit will have 3k-1 more qubits than the original circuit.
    Of these, 2k will be permanent data qubits, and k-1 will be ancilla
    qubits that disappear after each adder. I'm not sure how to notate that
    right now.
    '''

    def __init__(self, ks: list[int] = [3], 
                 add_measurements: bool = True,
                 use_catalyzed_sqrt_t: bool = True) -> None:
        '''

        Args:
            ks (list[int]): Will be the register sizes for the Adders used to perform
            RZ gates. Can perform rotation multiples of 2*pi/(2 ** k). If k=3,
            can perform T gates.

            add_measurements (bool): Whether or not to add Measurements and
            Resets to the circuit. This is useful for mappers, but not for
            unitary-based subroutines.

            use_catalyzed_sqrt_t (bool): For k = 4, we can use a catalyzed 
            sqrt(T) ancilla which requires 1 extra persistent ancilla. 
            However, we save on space and time as compared to the full PKAC.
        '''
        self.ks = ks
        self.add_measurements = add_measurements
        self.use_catalyzed_sqrt_t = use_catalyzed_sqrt_t

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
            if isinstance(op.gate, FractionalRZGate) and op.gate.k <= 3:
                # Replace with T gate TODO: fix with actual circuit
                circuit.replace_with_circuit(
                    CircuitPoint(cycle, op.location[0]),
                    FractionalRZGate.get_circuit(op.gate.numerator, op.gate.k),
                    as_circuit_gate=True
                )

        circuit.unfold_all()

    @staticmethod
    def convert_normal_rz(circuit: Circuit, skip_ks: list[int]) -> None:
        '''
        Convert all FractionalRZGates with k = 3 to T gates. This is just a
        special case of the general conversion, but it is useful to do this
        first since it allows us to use the T gate count as a proxy for how
        well the conversion worked (since T gates are expensive and we want to
        minimize them).
        '''
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, FractionalRZGate) and op.gate.k not in skip_ks:
                # Replace with T gate TODO: fix with actual circuit
                circuit.replace_gate(
                    CircuitPoint(cycle, op.location[0]),
                    RZGate(),
                    op.location,
                    [2 * np.pi * op.gate.numerator/(2 ** op.gate.k)]
                )

    async def run(self, circuit: Circuit, data: PassData) -> None:
        # New circuit size is num_qudits + k (input a) + sum(qft_state_sizes) input bs + k - 1 ancillas

        # We need to first convert all FractionalRZGates with k = 3
        # to circuits with T gates
        ks_to_skip = []
        if 3 in self.ks:
            self.convert_k3_to_t(circuit)
            self.ks.remove(3)
            ks_to_skip.append(3)
            
        extra_qubits = 0
        if 4 in self.ks and self.use_catalyzed_sqrt_t:
            # The last (non-ancilla) qubit will hold a sqrt(T) state
            # The other qubit will be storage for a second sqrt(T) state
            extra_qubits += 2
            ks_to_skip.append(4)
        
        # Now, convert all remaining FractionalRZGates with k not in self.ks
        # to RZ gates
        print("Skipping k values: ", ks_to_skip + self.ks, flush=True)
        print(circuit.gate_counts, flush=True)
        ConvertToPKAC.convert_normal_rz(circuit, skip_ks=self.ks + ks_to_skip)
        print("After converting normal RZs: ", circuit.gate_counts, flush=True)

        n = circuit.num_qudits

        if len(self.ks) > 0:
            # Yes PKAC circuit(s)
            max_k = max(self.ks)
            circuit_size = (circuit.num_qudits + max_k + sum(self.ks) 
                            + extra_qubits + max_k - 1)
        else:
            if extra_qubits == 0:
                 # No more decomps
                return
            # Require ancilla for logical AND
            circuit_size = circuit.num_qudits + extra_qubits + 1
            input_a_qubits = None

        new_circ = Circuit(circuit_size)

        # Register setup

        # Circuit Qubits -  0 -> n
        # Input A (length(k) if k > 5): n -> n + max_k
        # Input Bs (length(ks[i]) for each k): n + max_k -> n + max_k + sum(ks)
        # Extra qubits for catalyzed sqrt_t: n + max_k + sum(ks) -> n + max_k + sum(ks) 

        next_qubit = n
        input_a_qubits = None
        all_input_b_qubits = None
        ancilla_qubits = None

        if len(self.ks) > 0:
            input_a_qubits = list(range(n, n + max_k))
            next_qubit += max_k
            all_input_b_qubits = []
            start = next_qubit

            for size in self.ks:
                end = start + size
                input_b_qubits = list(range(start, end))
                all_input_b_qubits.append(input_b_qubits)
                # Initialize in QFT state
                new_circ.append_gate(XGate(), [input_b_qubits[-1]])
                new_circ.append_circuit(QFTGadget.generate(size), input_b_qubits)
                start = end

            next_qubit = start

        if self.use_catalyzed_sqrt_t:
            sqrt_t_qubit = next_qubit
            extra_sqrt_t_storage = next_qubit + 1
            # Initialize with sqrt(T) state -> Will be decomposed into HST later
            new_circ.append_gate(HGate(), [sqrt_t_qubit])
            new_circ.append_gate(RZGate(), (sqrt_t_qubit,), [np.pi / 8])
            next_qubit += 2
        else:
            sqrt_t_qubit = None
            extra_sqrt_t_storage = None

        # Remaining qubits are ancilla qubits
        ancilla_qubits = list(range(next_qubit, circuit_size))
        # Initialize ancilla qubits with measurements
        if self.add_measurements:
            for q in ancilla_qubits:
                new_circ.append_gate(
                    MidCircuitMeasurement('a'), [q],
                )

        for op in circuit.operations():
            if (isinstance(op.gate, FractionalRZGate) and op.gate.k == 4
                and self.use_catalyzed_sqrt_t):
                # Use cataylzed sqrt state
                # If numerator is even, then do num /2 and k = 3 decomp
                circ = FractionalRZGate.get_circuit(op.gate.numerator // 2, 3)
                new_circ.append_circuit(circ, [op.location[0]], as_circuit_gate=False)
                    
                if op.gate.numerator % 2 == 1:
                    # Apply sqrt(T) circuit
                    new_circ.append_circuit(
                        sqrt_circuit(),
                        [op.location[0], extra_sqrt_t_storage, 
                        sqrt_t_qubit, ancilla_qubits[0]],
                        as_circuit_gate=False
                    )

            elif isinstance(op.gate, FractionalRZGate):
                # Replace with adder circuit
                q = op.location[0]

                # Get correct k to use
                qft_ind = self.ks.index(op.gate.k)
                input_b_qubits = all_input_b_qubits[qft_ind]
                new_k = self.ks[qft_ind]

                # Add a CNOT to LSB on input A
                num = op.gate.numerator % (2 ** (new_k - 1))
                cnots = self.calculate_cnot_circuit(num, new_k)

                new_circ.append_circuit(cnots, [q] + input_a_qubits[:new_k])

                input_b = all_input_b_qubits[qft_ind]

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
