"""This module implements the GidneyAdder."""
from __future__ import annotations

from typing import TYPE_CHECKING

from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.constant.t import TGate

from bqskit.ft.gates.logical_and import LogicalAndGate, LogicalAndDgGate


from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.reset import Reset

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
    def __init__(self, register_size: int) -> None:
        """

        Args:
            circuit (Circuit): The circuit to copy into gate format.

            move (bool): If true, the constructor will not copy the circuit.
                This should only be used when you are sure `circuit` will no
                longer be used on caller side. If unsure use the default.
                (Default: False)
        """

        self._circuit = self.generate_circuit(register_size)
        self._num_qudits = 3 * register_size - 1
        assert self._circuit.num_qudits == self._num_qudits
        self._radixes = self._circuit.radixes
        self._num_params = self._circuit.num_params
        self._name = 'GidneyAdder(%s)' % str(register_size)

    def generate_circuit(self, n: int) -> Circuit:
        '''
        Generate an adder from logical ands. Uses 3n - 1 qubits.

        Takes |a, b, 0> to |a, a + b, 0>
        '''
        c = Circuit(3 * n - 1)
        # Initialize all ancilla in T state
        for i in range(2*n, 3*n - 1):
            c.append_gate(Reset(), [i])
            c.append_gate(HGate(), [i])
            c.append_gate(TGate(), [i])

        for i in range(n - 1):
            A_i = n - i - 1 # Indexing from MSB to LSB
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
            A_i = n - i - 1 # Indexing from MSB to LSB
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
            A_i = n - i - 1 # Indexing from MSB to LSB
            B_i = 2 * n - i - 1 
            c.append_gate(CNOTGate(), [A_i, B_i])

        return c