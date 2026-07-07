"""Generate a quantum Fourier transform that can be applied to some qubits."""
from __future__ import annotations

from numpy import pi

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import SwapGate
from bqskit.ir.gates import TdgGate
from bqskit.ir.gates import TGate


class QFTGadget:

    @staticmethod
    def controlled_rz_layer(width: int) -> Circuit:
        """
        A layer of controlled RZ gates decomposed in to CNOTs and RZs
        """
        assert width >= 2, 'Width must be at least 2 for a controlled RZ layer.'
        circuit = Circuit(width)
        for i in range(1, width):
            base = 2 ** (i + 1)
            if base == 4:
                gate_0, gate_i = TdgGate(), TGate()
                params_0, params_i = [], []
            else:
                gate_0, gate_i = RZGate(), RZGate()
                params_0, params_i = [-pi / base], [pi / base]
            circuit.append_gate(CNOTGate(), [i, 0])
            circuit.append_gate(gate_0, [0], params_0)
            circuit.append_gate(CNOTGate(), [i, 0])
            circuit.append_gate(gate_i, [i], params_i)
            circuit.append_gate(gate_i, [0], params_i)
        return circuit

    @staticmethod
    def generate(num_qudits: int) -> Circuit:
        """
        Generate a QFT that can be applied to qubits.

        Args:
            num_qudits (int): The number of qubits the QFT should be applied to.

        Returns:
            (CircuitGate): A QFT block that can be applied to qubits.
        """
        circuit = Circuit(num_qudits)
        for i in range(num_qudits - 1):
            circuit.append_gate(HGate(), [i])
            width = num_qudits - i
            subcirc = QFTGadget.controlled_rz_layer(width)
            support = list(range(i, num_qudits))
            circuit.append_circuit(subcirc, support)
        circuit.append_gate(HGate(), [num_qudits - 1])

        # Swap qubits at the end
        for i in range(num_qudits // 2):
            circuit.append_gate(SwapGate(), [i, num_qudits - i - 1])

        return circuit


if __name__ == '__main__':
    import numpy as np

    num_qudits = 8
    circuit = Circuit(num_qudits)
    qft = QFTGadget.generate(num_qudits)
    circuit.append_circuit(qft, [_ for _ in range(num_qudits)])
    circuit.unfold_all()
    for op in circuit:
        print(op, op.params)

    def qft_u(n):
        # this is the qft unitary generator code from qsearch
        root = np.e ** (2j * np.pi / n)
        return np.fromfunction(lambda x, y: root**(x * y), (n, n)) / np.sqrt(n)

    print('=' * 80)
    one = np.zeros((2 ** num_qudits, 2 ** num_qudits), dtype=complex)
    u_qft = qft_u(2 ** num_qudits)
    dist = circuit.get_unitary().get_distance_from(u_qft)
    print(dist)
