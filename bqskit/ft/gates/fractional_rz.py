"""This module implements the FractionalRZGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class FractionalRZGate(QubitGate, CachedClass):
    """
    A gate representing an arbitrary rotation around the Z axis.

    Takes 2 parameters, a numerator and k and implements the
    gate RZ(2pi * numerator / 2 ** k). This is useful for implementing
    the gates that can be implemented with PKAC.
    """

    _num_qudits = 1
    _num_params = 2
    _qasm_name = 'fractional_rz'

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)
        assert int(params[0]) == params[0], "Numerator must be an integer"
        assert int(params[1]) == params[1], "Denominator must be an integer"
        assert params[0] < 2 ** params[1], "Numerator must be less than 2^k"

        angle = 2 * np.pi * params[0] / (2 ** params[1])

        pexp = np.exp(1j * angle / 2)
        nexp = np.exp(-1j * angle / 2)

        return UnitaryMatrix(
            [
                [nexp, 0],
                [0, pexp],
            ],
        )