from __future__ import annotations

import numpy as np
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate

from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.cz import CZGate
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.constant.identity import IdentityGate
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.ir.gates.constant.sdg import SdgGate
from bqskit.ir.gates.constant.s import SGate
from bqskit.ir.gates.constant.sx import SqrtXGate
from bqskit.ir.gates.constant.swap import SwapGate
from bqskit.ir.gates.constant.tdg import TdgGate
from bqskit.ir.gates.constant.t import TGate
from bqskit.ir.gates.constant.x import XGate
from bqskit.ir.gates.constant.y import YGate
from bqskit.ir.gates.constant.z import ZGate


clifford_gates = [
    CNOTGate(),
    CZGate(),
    HGate(),
    SdgGate(),
    SGate(),
    SqrtXGate(),
    SwapGate(),
    XGate(),
    YGate(),
    ZGate(),
]

t_gates = [TGate(), TdgGate()]

rz_gates = [RZGate()]


def round_to_discrete_z(val: float, period: float) -> CircuitGate:
    """
    Returns a CircuitGate of discrete Z-axis rotations given a rounded `val`.

    Args:
        val (float): A Z-axis rotation value in radians to round.

        period (float): The period of the rounding function in multiples of
            pi. For example, if period is 0.5, the function will round to
            the nearest multiple of pi/2.

    Returns:
        (CircuitGate): A CircuitGate containing a Z rotation gate depending
            on val.
    """
    circuit = Circuit(1)
    rounded_val = int(np.round(val / np.pi / period))
    if period == 0.5:
        # cliffords
        rounded_val = rounded_val % 4
        if rounded_val == 1:
            circuit.append_gate(SGate(), 0)
        elif rounded_val == 2:
            circuit.append_gate(ZGate(), 0)
        elif rounded_val == 3:
            circuit.append_gate(SdgGate(), 0)
        elif rounded_val == 0:
            circuit.append_gate(IdentityGate(), 0)
    elif period == 0.25:
        rounded_val = rounded_val % 8
        if rounded_val == 0:
            circuit.append_gate(IdentityGate(), 0)
        elif rounded_val < 4:
            if rounded_val >= 2:
                circuit.append_gate(SGate(), 0)
            if rounded_val % 2:
                circuit.append_gate(TGate(), 0)
        elif rounded_val > 4:
            if rounded_val <= 6:
                circuit.append_gate(SdgGate(), 0)
            if rounded_val % 2:
                circuit.append_gate(TdgGate(), 0)
        elif rounded_val == 4:
            circuit.append_gate(ZGate(), 0)
    return CircuitGate(circuit)


def circuit_for_rounded_val(val: float, period: float) -> CircuitGate:
    return round_to_discrete_z(val, period)
