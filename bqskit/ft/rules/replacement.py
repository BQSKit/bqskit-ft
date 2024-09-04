from __future__ import annotations

from functools import partial
from typing import Callable

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.circuit import CircuitGate
from bqskit.ir.circuit import Gate
from bqskit.ir.operation import Operation
from bqskit.qis.unitary import UnitaryMatrix


def construct_unitary_match_rule(
    unitary: UnitaryMatrix,
    threshold: float = 1e-8,
) -> Callable[[Operation], bool]:
    return partial(
        unitary_match_function,
        unitary,
        threshold,
    )


def unitary_match_function(
    unitary: UnitaryMatrix,
    threshold: float,
    op: Operation,
) -> bool:
    return op.get_unitary().get_distance_from(unitary) < threshold


class ReplacementRule(BasePass):

    def __init__(
        self,
        indicator: Callable[[Circuit | Operation], bool],
        replacement: Circuit | CircuitGate | Gate,
    ) -> None:
        """
        Replace a partition with `replacement` if `indicator` is True.

        Args:
            indicator (Callable[[Circuit | Operation], bool]): The function
                to determine if the replacement should be applied.

            replacement (Circuit | CircuitGate | Gate): The Circuit or Gate
                to replace the partition with. The width of the replacement
                must match the partition's width.
        """
        self.indicator = indicator
        if isinstance(replacement, Gate):
            num_qudits = replacement.num_qudits
            self.replacement = Circuit(num_qudits, replacement.radixes)
            self.replacement.append_gate(
                replacement, [_ for _ in range(num_qudits)],
            )
            self.replacement.unfold_all()
        else:
            self.replacement = replacement

    async def run(self, circuit: Circuit, data: PassData) -> None:
        if circuit.num_qudits != self.replacement.num_qudits:
            return
        replace = self.indicator(circuit)
        if replace:
            circuit.become(self.replacement, deepcopy=True)
