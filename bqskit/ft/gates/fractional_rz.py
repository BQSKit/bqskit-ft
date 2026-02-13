"""This module implements the FractionalRZGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate


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
        self.numerator = numerator
        self.k = k
        assert int(numerator) == numerator, "Numerator must be an integer"
        assert int(k) == k, "k must be an integer"
        assert numerator < 2 ** k, "Numerator must be less than 2^k"


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