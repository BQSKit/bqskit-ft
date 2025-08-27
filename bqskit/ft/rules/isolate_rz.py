"""This module implements the IsolateRZGatePass."""
from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import MeasurementPlaceholder
from bqskit.ir.gates import Reset
from bqskit.ir.gates.barrier import BarrierPlaceholder
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.ir.region import CircuitRegion


class IsolateRZGatePass(BasePass):
    """
    The IsolateRZGatePass Pass.

    This pass forms CircuitGates around RZ gates.
    """

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        # Go through each qudit individually
        for q in range(circuit.num_qudits):

            single_qubit_regions = []

            for c in range(circuit.num_cycles):
                if circuit.is_point_idle((c, q)):
                    continue

                op = circuit[c, q]
                if (
                    op.num_qudits == 1
                    and not isinstance(
                        op.gate, (
                            BarrierPlaceholder,
                            MeasurementPlaceholder,
                            Reset,
                        ),
                    )
                ) and (isinstance(op.gate, RZGate)):
                    region = CircuitRegion({q: (c, c)})
                    single_qubit_regions.append(region)

            for region in reversed(single_qubit_regions):
                circuit.fold(region)
