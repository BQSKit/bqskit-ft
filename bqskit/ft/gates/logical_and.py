"""This module implements the LogicalAndGate and LogicalAndDgGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.constant.s import SGate
from bqskit.ir.gates.constant.t import TGate
from bqskit.ir.gates.constant.tdg import TdgGate
from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class LogicalAndGate(CircuitGate):
    """
    The LogicalAndGate from Gidney's "Halving the cost of quantum addition"
    paper.

    https://arxiv.org/pdf/1709.06648

    This gate takes three qubits as input. Takes |a,b,T> to |a,b,a AND b>.

    Will not work if the ancilla is not in the T state,
    so it is not a general purpose gate.
    """

    def __init__(self) -> None:
        """

        Args:
            circuit (Circuit): The circuit to copy into gate format.

            move (bool): If true, the constructor will not copy the circuit.
                This should only be used when you are sure `circuit` will no
                longer be used on caller side. If unsure use the default.
                (Default: False)
        """

        self._circuit = self.generate_circuit()
        self._num_qudits = 3
        assert self._circuit.num_qudits == self._num_qudits
        self._radixes = self._circuit.radixes
        self._num_params = self._circuit.num_params
        self._name = 'LogicalAndGate'

    def generate_circuit(self) -> Circuit:
        '''
        Generate an adder from logical ands. Uses 3n - 1 qubits.

        Takes |a,b,T> to |a,b,a AND b>
        '''
        c = Circuit(3)
        c.append_gate(CNOTGate(), [0, 2])
        c.append_gate(CNOTGate(), [1, 2])
        c.append_gate(CNOTGate(), [2, 0])
        c.append_gate(CNOTGate(), [2, 1])
        c.append_gate(TdgGate(), [0])
        c.append_gate(TdgGate(), [1])
        c.append_gate(TGate(), [2])
        c.append_gate(CNOTGate(), [2, 1])
        c.append_gate(CNOTGate(), [2, 0])
        c.append_gate(HGate(), [2])
        c.append_gate(SGate(), [2])
        return c


class LogicalAndDgGate(ConstantGate, QubitGate):
    """
    The inverse of the LogicalAndGate.

    Takes |a,b,a AND b> to |a,b,0>.
    """

    def __init__(self) -> None:
        '''
        Note that this gate decomposes to a measurement and classical
        fix up, which is not currently supported in BQSKit. So we implement
        the unitary directly on the 3 qubits.
        '''

        # Create unitary as a function on the basis states.
        U = np.zeros((8, 8), dtype=complex)
        # Sum over all basis inds
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    input_state = (a << 2) | (b << 1) | c
                    if c == (a & b):
                        output_state = (a << 2) | (b << 1) | 0
                    else:
                        output_state = (a << 2) | (b << 1) | 1
                    U[output_state, input_state] = 1

        U = UnitaryMatrix(U)
        self._utry = U
        self._num_qudits = self._utry.num_qudits
        self._radixes = self._utry.radixes
