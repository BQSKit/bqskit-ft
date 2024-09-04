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