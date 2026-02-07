"""This module implements the ZXZXZDecomposition."""
from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ft.gates.sxdg import SXdgGate
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.constant.sx import SXGate
from bqskit.ir.gates.parameterized.rx import RXGate
from bqskit.ir.gates.parameterized.ry import RYGate
from bqskit.ir.gates.parameterized.rz import RZGate


class XYtoZRotation(BasePass):
    """
    The XYtoZRotation class.

    Express an RX or RY rotation circuit with an RZ rotation.
    """
    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        if circuit.num_qudits != 1:
            m = 'Cannot convert a multi-qudit circuit into a sequence of '
            m += 'Clifford * RZ * Clifford gates.'
            raise ValueError(m)

        if circuit.radixes[0] != 2:
            m = 'Cannot convert a non-qubit circuit into a sequence of '
            m += 'Clifford * RZ * Clifford gates.'
            raise ValueError(m)

        contains_one_gate = circuit.num_operations == 1
        contains_rx = RXGate() in circuit.gate_set
        contains_ry = RYGate() in circuit.gate_set
        contains_rz = RZGate() in circuit.gate_set

        if not contains_one_gate:
            m = 'Circuit must contain exactly one operation.'
            raise ValueError(m)

        if not (contains_rx or contains_ry or contains_rz):
            m = 'Circuit must contain an RX, RY, or RZ gate'
            raise ValueError(m)

        p = circuit.params[0]

        new_circuit = Circuit(1)
        if RXGate() in circuit.gate_set:
            new_circuit.append_gate(HGate(), 0)
            new_circuit.append_gate(RZGate(), 0, [p])
            new_circuit.append_gate(HGate(), 0)
        elif RYGate() in circuit.gate_set:
            new_circuit.append_gate(SXGate(), 0)
            new_circuit.append_gate(RZGate(), 0, [p])
            new_circuit.append_gate(SXdgGate(), 0)
        elif RZGate() in circuit.gate_set:
            new_circuit.append_gate(RZGate(), 0, [p])

        circuit.become(new_circuit)
