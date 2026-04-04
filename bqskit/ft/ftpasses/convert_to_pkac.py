from __future__ import annotations
import numpy as np

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ft.gadgets.phase_gradient import PhaseGradientGadget
from bqskit.ft.gates.fractional_rz import FractionalRZGate
from bqskit.ft.gates.gidney_adder import ConstantGidneyAdder
from bqskit.ft.gates.logical_and import LogicalAndDgGate, LogicalAndGate
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.constant.t import TGate
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

    def __init__(self, max_k: int = None, 
                 add_measurements: bool = True,
                 use_catalyzed_sqrt_t: bool = True) -> None:
        '''

        Args:
            max_k (int): The maximum value of k for which to generate PKAC circuits.
            add_measurements (bool): Whether or not to add Measurements and
            Resets to the circuit. This is useful for mappers, but not for
            unitary-based subroutines.

            use_catalyzed_sqrt_t (bool): For k = 4, we can use a catalyzed 
            sqrt(T) ancilla which requires 1 extra persistent ancilla. 
            However, we save on space and time as compared to the full PKAC.
        '''
        self.max_k = max_k
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
    def convert_normal_rz(circuit: Circuit, skip_max: int) -> None:
        '''
        Convert all FractionalRZGates with k = 3 to T gates. This is just a
        special case of the general conversion, but it is useful to do this
        first since it allows us to use the T gate count as a proxy for how
        well the conversion worked (since T gates are expensive and we want to
        minimize them).
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
        # We need to first convert all FractionalRZGates with k = 3
        # to circuits with T gates
        has_k_4 = False
        # Use largest k over all FractionalRZGates in the circuit
        max_k_used = 0
        for op in circuit.operations():
            if isinstance(op.gate, FractionalRZGate):
                k = op.gate.k
                if k > max_k_used:
                    max_k_used = k
                if k == 4:
                    has_k_4 = True

        if self.max_k is None:
            self.max_k = max_k_used

        # print("Before converting to PKAC: ", circuit.gate_counts, flush=True)
        # print("Max k used: ", self.max_k, flush=True)

        self.convert_k3_to_t(circuit)
            
        extra_qubits = 0
        if has_k_4 and self.use_catalyzed_sqrt_t:
            # The last (non-ancilla) qubit will hold a sqrt(T) state
            # The other qubit will be storage for a second sqrt(T) state
            extra_qubits += 2
        
        # Now, convert all remaining FractionalRZGates to RZ gates
        ConvertToPKAC.convert_normal_rz(circuit, skip_max=self.max_k)
        # print("After converting normal RZs: ", circuit.gate_counts, flush=True)

        n = circuit.num_qudits

        if self.max_k > 4 or (has_k_4 and not self.use_catalyzed_sqrt_t):
            do_pkac = True
            # max_k registers for phase_gradient, and max_k - 1 ancillas
            circuit_size = (circuit.num_qudits + 2*self.max_k - 1 + extra_qubits)
        else:
            do_pkac = False
            if extra_qubits == 0:
                 # No more decomps
                return
            # Require extra ancilla for logical AND
            circuit_size = circuit.num_qudits + extra_qubits + 1

        new_circ = Circuit(circuit_size)

        # Register setup

        # Circuit Qubits -  0 -> n
        # Input A (length(k) if k > 5): n -> n + max_k
        # Input Bs (length(ks[i]) for each k): n + max_k -> n + max_k + sum(ks)
        # Extra qubits for catalyzed sqrt_t: n + max_k + sum(ks) -> n + max_k + sum(ks) 

        next_qubit = n
        input_b_qubits = None
        ancilla_qubits = None

        if do_pkac:
            input_b_qubits = list(range(next_qubit, next_qubit + self.max_k))
            new_circ.append_circuit(PhaseGradientGadget.generate(self.max_k), 
                                    input_b_qubits)
            next_qubit += self.max_k

        if self.use_catalyzed_sqrt_t and has_k_4:
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
                new_circ.append_circuit(circ, 
                                        [op.location[0]], 
                                        as_circuit_gate=False)
                    
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

                # Add a CNOT to LSB on input A
                # num = op.gate.numerator % (2 ** (self.max_k - 1))
                # Convert num / 2 ** (op.gate.k - 1) to new_num / 2 ** (self.max_k - 1)
                diff = self.max_k - op.gate.k
                num = op.gate.numerator * (2 ** diff) % (2 ** (self.max_k - 1))

                # Kickback from adder with CNOTs
                for i in range(self.max_k):
                    new_circ.append_gate(CNOTGate(), [q, input_b_qubits[i]])

                gidney_adder = ConstantGidneyAdder(self.max_k, num, add_reset=self.add_measurements)
                gidney_loc = (input_b_qubits + ancilla_qubits[:self.max_k - 1])

                # Apply adder on A, B, and ancilla
                new_circ.append_gate(
                    gidney_adder,
                    gidney_loc
                )

                # Kickback from adder with CNOTs
                for i in range(self.max_k):
                    new_circ.append_gate(CNOTGate(), [q, input_b_qubits[i]])
            else:
                new_circ.append(op)
        circuit.become(new_circ)
