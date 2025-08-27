"""Apply the gridsynth algorithm to an RZ gate."""
from __future__ import annotations

import mpmath
from pygridsynth.gridsynth import gridsynth_gates

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.constant.s import SGate
from bqskit.ir.gates.constant.t import TGate
from bqskit.ir.gates.constant.x import XGate
from bqskit.ir.gates.parameterized.rz import RZGate
mpmath.mp.dps = 128


class GridSynthPass(BasePass):
    def __init__(
        self,
        precision: int = 10,
    ) -> None:
        self.precision = mpmath.mpmathify(f'1e-{precision}')

    async def run(self, circuit: Circuit, data: PassData) -> None:
        if circuit.num_qudits != 1:
            m = 'GridSynthPass only works on single qubit inputs. '
            m = f'Got an input with {circuit.num_qudits} qudits.'
            raise ValueError(m)

        if circuit.gate_counts.get(RZGate(), 0) != 1:
            m = 'The input must be a single RZ gate. '
            m += f'Got an input with {circuit.gate_counts}.'
            raise ValueError(m)

        epsilon = data['precision'] if 'precision' in data else self.precision
        theta = mpmath.mpmathify(circuit.params[0])
        gates = gridsynth_gates(theta, epsilon)

        new_circuit = Circuit(1)
        # Gates are in matrix order, so we need to append them in reverse
        for gate in reversed(gates):
            if gate == 'H':
                new_circuit.append_gate(HGate(), [0])
            elif gate == 'X':
                new_circuit.append_gate(XGate(), [0])
            elif gate == 'S':
                new_circuit.append_gate(SGate(), [0])
            elif gate == 'T':
                new_circuit.append_gate(TGate(), [0])

        circuit.become(new_circuit)
