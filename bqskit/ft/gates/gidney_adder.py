"""This module implements the GidneyAdder."""
from __future__ import annotations

from bqskit.ft.gates.logical_and import LogicalAndDgGate
from bqskit.ft.gates.logical_and import LogicalAndGate
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.constant.t import TGate
from bqskit.ir.gates.constant.x import XGate
from bqskit.ir.gates.measure import MidCircuitMeasurement
from bqskit.ir.gates.reset import Reset


class ConstantGidneyAdder(CircuitGate):
    """
    The Adder from Gidney's "Halving the cost of quantum addition" paper.

    This adds a classical constant to a quantum register.

    https://arxiv.org/pdf/1709.06648

    This adder takes one input register of size n and a constant value. 
    It creates a gate with 2n - 1 qubits.

    Qubits 0 to n - 1 are input A, qubits n to 2n - 1 are input B,
    and qubits 2n to 3n - 2 are ancilla. The 0th qubit in each register
    is the MSB, as is standard in BQSKit.

    """

    def __init__(self, 
                 register_size: int, 
                 constant: int, 
                 add_reset: bool = True,
                 msb_plus: bool = True) -> None:
        """

        Args:
            register_size (int) : The size of the inputs A and B passed to the
            adder. The total circuit will have width 3*(`register_size`) - 1.
            Inputs A and B will be of size `register_size` and there will be an
            addition `register_size` - 1 ancilla for T gates.

            constant (int): The constant to add to the input register. 
            Must be less than 2^`register_size`.

            add_reset (bool): Whether or not to add the reset gate to the
            circuit. A reset gate is useful to pass to mappers, but will throw
            an error for unitary-based subroutines.

            msb_plus (bool): Whether the most significant bit of the input is
            in the plus state. This is the case if input B is the phase
            gradient state.
        """

        self._circuit = self.generate_circuit(register_size, constant, add_reset)
        self._num_qudits = 2 * register_size - 1
        assert self._circuit.num_qudits == self._num_qudits
        self._radixes = self._circuit.radixes
        self._num_params = self._circuit.num_params
        self.msb_plus = msb_plus
        self._name = 'ConstantGidneyAdder(%s, %s)' % (str(register_size), str(constant))

    def generate_circuit(self, n: int, constant: int, add_reset: bool = True) -> Circuit:
        '''
        Generate an adder from logical ands. Uses 3n - 1 qubits.

        Takes |b, 0> to |b + constant, 0>
        '''
        c = Circuit(2 * n - 1)

        # B 0 -> n - 1, Ancilla n -> 2n - 2

        # Convert constant to bits with LSB first
        constant_bits = [(constant >> i) & 1 for i in range(n)]

        # Start circuit with first ind that is not 0
        start_ind = 0
        while start_ind < n and constant_bits[start_ind] == 0:
            start_ind += 1

        # Initialize all ancilla in T state
        for q  in range(n + start_ind + 1, 2 * n - 1):
            if add_reset:
                c.append_gate(Reset(), [q])
            c.append_gate(HGate(), [q])
            c.append_gate(TGate(), [q])

        for i in range(start_ind, n - 1):
            # Using circuit defined here: https://arxiv.org/pdf/2007.07391#section*.51
            # Figure 18
            # Get appropriate qubits
            B_i = n - i - 1
            anc_i = 2 * n - 2 - i
            B_i_plus_1 = n - i - 2
            anc_i_minus_1 = 2 * n - 1 - i
            # Create logical and circuit
            # And bits A[i], B[i] and Anc[i]
            if i == start_ind:
                # For the first bit, no need to consider previous and
                # Logical and just becomes a single CNOT, no T state needed
                if constant_bits[i] == 1:
                    c.append_gate(CNOTGate(), [B_i, anc_i])
            else: # i > 0
                if constant_bits[i] == 1:
                    c.append_gate(XGate(), [anc_i_minus_1])
                c.append_gate(LogicalAndGate(), [anc_i_minus_1, B_i, anc_i])
                if constant_bits[i] == 1:
                    c.append_gate(XGate(), [anc_i_minus_1])
                c.append_gate(CNOTGate(), [anc_i_minus_1, anc_i])

            
            # Waterfall and output to next A and B
            c.append_gate(CNOTGate(), [anc_i, B_i_plus_1])


        # Now perform the inverse
        for i in reversed(range(n - 1)):
            # Get appropriate qubits
            B_i = n - i - 1
            anc_i = 2 * n - 2 - i
            B_i_plus_1 = n - i - 2
            anc_i_minus_1 = 2 * n - 1 - i

            if i < n - 2 and i > start_ind:
                # Reverse waterfall
                c.append_gate(CNOTGate(), [anc_i, B_i_plus_1])

            if i > start_ind:
                if constant_bits[i] == 1:
                    c.append_gate(XGate(), [anc_i_minus_1])
                c.append_gate(LogicalAndDgGate(), [anc_i_minus_1, B_i, anc_i])
                if constant_bits[i] == 1:
                    c.append_gate(XGate(), [anc_i_minus_1])
            else: # i == start_ind
                if constant_bits[i] == 1:
                    c.append_gate(CNOTGate(), [B_i, anc_i])
                if add_reset:
                    c.append_gate(MidCircuitMeasurement('a'), [anc_i])
                    c.append_gate(Reset(), [anc_i])

        # Final layer of X gatess
        for i in range(n):
            if constant_bits[i] == 1:
                c.append_gate(XGate(), [B_i])

        return c


class GidneyAdder(CircuitGate):
    """
    The Adder from Gidney's "Halving the cost of quantum addition" paper.

    https://arxiv.org/pdf/1709.06648

    This adder takes one input register of size n. It creates a gate
    with 3n - 1 qubits.

    Qubits 0 to n - 1 are input A, qubits n to 2n - 1 are input B,
    and qubits 2n to 3n - 2 are ancilla. The 0th qubit in each register
    is the MSB, as is standard in BQSKit.

    """

    def __init__(self, register_size: int, add_reset: bool = True) -> None:
        """

        Args:
            register_size (int) : The size of the inputs A and B passed to the
            adder. The total circuit will have width 3*(`register_size`) - 1.
            Inputs A and B will be of size `register_size` and there will be an
            addition `register_size` - 1 ancilla for T gates.

            add_reset (bool): Whether or not to add the reset gate to the
            circuit. A reset gate is useful to pass to mappers, but will throw
            an error for unitary-based subroutines.
        """

        self._circuit = self.generate_circuit(register_size, add_reset)
        self._num_qudits = 3 * register_size - 1
        assert self._circuit.num_qudits == self._num_qudits
        self._radixes = self._circuit.radixes
        self._num_params = self._circuit.num_params
        self._name = 'GidneyAdder(%s)' % str(register_size)

    def generate_circuit(self, n: int, add_reset: bool = True) -> Circuit:
        '''
        Generate an adder from logical ands. Uses 3n - 1 qubits.

        Takes |a, b, 0> to |a, a + b, 0>
        '''
        c = Circuit(3 * n - 1)
        # Initialize all ancilla in T state
        for i in range(2 * n, 3 * n - 1):
            if add_reset:
                c.append_gate(Reset(), [i])
            c.append_gate(HGate(), [i])
            c.append_gate(TGate(), [i])

        for i in range(n - 1):
            A_i = n - i - 1  # Indexing from MSB to LSB
            B_i = 2 * n - i - 1
            anc_i = 3 * n - 2 - i
            A_i_plus_1 = n - i - 2
            B_i_plus_1 = 2 * n - i - 2
            anc_i_minus_1 = 3 * n - 1 - i
            # Create logical and circuit
            # And bits A[i], B[i] and Anc[i]
            c.append_gate(LogicalAndGate(), [A_i, B_i, anc_i])
            # Water fall and to next A[i + 1], B[i + 1]
            if i > 0:
                # CNOT from previous and output to current and output
                c.append_gate(CNOTGate(), [anc_i_minus_1, anc_i])

            if i < n - 2:
                # Waterfall and output to next A and B
                c.append_gate(CNOTGate(), [anc_i, A_i_plus_1])
                c.append_gate(CNOTGate(), [anc_i, B_i_plus_1])
            elif i == n - 2:
                # For the last bit, we need CNOT and output with B[i + 1] only
                c.append_gate(CNOTGate(), [anc_i, B_i_plus_1])

        # Now perform the inverse
        for i in reversed(range(n - 1)):
            A_i = n - i - 1  # Indexing from MSB to LSB
            B_i = 2 * n - i - 1
            anc_i = 3 * n - 2 - i
            A_i_plus_1 = n - i - 2
            anc_i_minus_1 = 3 * n - 1 - i
            if i < n - 2:
                # Only 1 CNOT from Anc[i] to A[i + 1]
                c.append_gate(CNOTGate(), [anc_i, A_i_plus_1])
            if i > 0:
                # CNOT from previous and output to current and output
                c.append_gate(CNOTGate(), [anc_i_minus_1, anc_i])

            c.append_gate(LogicalAndDgGate(), [A_i, B_i, anc_i])

        # Final layer of CNOTs
        for i in range(n):
            A_i = n - i - 1  # Indexing from MSB to LSB
            B_i = 2 * n - i - 1
            c.append_gate(CNOTGate(), [A_i, B_i])

        return c
