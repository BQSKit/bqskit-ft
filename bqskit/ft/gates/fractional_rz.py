"""This module implements the FractionalRZGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.s import SGate
from bqskit.ir.gates.constant.t import TGate
from bqskit.ir.gates.constant.z import ZGate
from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

class FractionalRZGate(ConstantGate, QubitGate):
    """
    A gate representing an arbitrary rotation around the Z axis.

    Takes 2 parameters, a numerator and k and implements the
    gate RZ(2pi * numerator / 2 ** k). This is useful for implementing
    the gates that can be implemented with PKAC.
    """

    _num_qudits = 1
    _num_params = 0
    _qasm_name = 'fractional_rz'

    def __init__(self, numerator: int = 0, k: int = 0) -> None:
        self.numerator = numerator % (2 ** k)
        self.k = k
        assert int(numerator) == numerator, 'Numerator must be an integer'
        assert int(k) == k, 'k must be an integer'

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        angle = 2 * np.pi * self.numerator / (2 ** self.k)

        pexp = np.exp(1j * angle / 2)
        nexp = np.exp(-1j * angle / 2)

        return UnitaryMatrix(
            [
                [nexp, 0],
                [0, pexp],
            ],
        )
    
    @staticmethod
    def get_circuit(numerator: int, k: int) -> Circuit:
        """Return a circuit implementing this gate."""
        circ = Circuit(1)
        if k > 3:
            circ.append_gate(FractionalRZGate(numerator, k), [0])
        else:
            # 2p * num / 8
            pi_angle = (2 * numerator / (2 ** k)) % 2
            # Fix angle between 0 and 2pi
            while pi_angle >= 1: # pi rotations
                pi_angle -= 1
                circ.append_gate(ZGate(), [0])
            while pi_angle >= 0.5: # pi/2 rotations
                pi_angle -= 0.5
                circ.append_gate(SGate(), [0])
            if pi_angle > 0: # pi/4 rotations
                circ.append_gate(TGate(), [0])
                pi_angle -= 0.25
        return circ
