"""Apply the gridsynth algorithm to an RZ gate."""
from __future__ import annotations

import mpmath
from pygridsynth.gridsynth import gridsynth_gates

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.operation import Operation
from bqskit.ir.circuit import CircuitGate
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.constant.s import SGate
from bqskit.ir.gates.constant.t import TGate
from bqskit.ir.gates.constant.x import XGate
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.runtime import get_runtime
mpmath.mp.dps = 128


class GridSynthPass(BasePass):
    def __init__(
        self,
        algorithmic_error: float = 1e-4,
    ) -> None:
        self.error = algorithmic_error


    async def run_rz(self, theta: float, precision: float) -> Circuit:
        """Run the gridsynth algorithm on a single RZ gate."""
        theta = mpmath.mpmathify(theta)
        precision = mpmath.mpmathify(precision)
        gate_str = gridsynth_gates(theta, precision)
        new_circuit = Circuit(1)
        # Gates are in matrix order, so we need to append them in reverse
        for gate in reversed(gate_str):
            if gate == 'H':
                new_circuit.append_gate(HGate(), [0])
            elif gate == 'X':
                new_circuit.append_gate(XGate(), [0])
            elif gate == 'S':
                new_circuit.append_gate(SGate(), [0])
            elif gate == 'T':
                new_circuit.append_gate(TGate(), [0])

        return new_circuit

    async def run(self, circuit: Circuit, data: PassData) -> None:
        num_rzs = circuit.num_params
        assert circuit.count(RZGate()) == num_rzs, "Circuit has {} RZ gates but {} parameters".format(
            circuit.count(RZGate()), num_rzs
        )

        if num_rzs == 0:
            return

        gg_circs = await get_runtime().map(
            self.run_rz,
            circuit.params,
            precision = self.error/num_rzs,
        )

        # Batch replace the RZ gates with the new circuits
        rz_pts = []
        gg_ops = []
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, RZGate):
                rz_pts.append((cycle, op.location[0]))
                gg_ops.append(Operation(
                    CircuitGate(gg_circs[len(rz_pts) - 1]),
                    op.location,
                ))

        circuit.batch_replace(
            rz_pts,
            gg_ops,
        )

        circuit.unfold_all()



