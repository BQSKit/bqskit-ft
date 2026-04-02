"""Generate a quantum Fourier transform that can be applied to some qubits."""
from __future__ import annotations

from numpy import pi

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import HGate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import TGate
from bqskit.ir.gates.constant.s import SGate
from bqskit.ir.gates.constant.z import ZGate


class PhaseGradientGadget:

    @staticmethod
    def generate(num_qudits: int) -> Circuit:
        """
        Generate a Phase Gradient Register that can be used.

        Args:
            num_qudits (int): The number of qubits the QFT should be applied to.

        Returns:
            (CircuitGate): A QFT block that can be applied to qubits.
        """
        circuit = Circuit(num_qudits)
        for i in range(num_qudits):
            circuit.append_gate(HGate(), [i])
            if i == 0:
                circuit.append_gate(ZGate(), [i])
            if i == 1:
                circuit.append_gate(SGate(), [i])
            if i == 2:
                circuit.append_gate(TGate(), [i])
            if i > 2:
                circuit.append_gate(RZGate(), [i], [pi / 2 ** (i)])

        return circuit