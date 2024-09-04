from __future__ import annotations

from numpy import pi
from numpy import round

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.operation import Operation

from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.ir.gates.constant.sdg import SdgGate
from bqskit.ir.gates.constant.s import SGate
from bqskit.ir.gates.constant.tdg import TdgGate
from bqskit.ir.gates.constant.t import TGate
from bqskit.ir.gates.constant.z import ZGate
from bqskit.ir.gates.constant.identity import IdentityGate


class RoundToDiscreteZPass(BasePass):

    def __init__(self, synthesis_epsilon: float = 1e-8) -> None:
        self.synthesis_epsilon = synthesis_epsilon

    def check_angle(self, angle: float) -> Circuit | None:
        value = int(round(4 * angle / pi)) % 8
        rounded_angle = value * pi / 2 % (2 * pi)
        residual = angle % (2 * pi) - rounded_angle

        if residual > self.synthesis_epsilon:
            return None

        if value == 0:
            gates = [IdentityGate()]
        elif value == 1:
            gates = [TGate()]
        elif value == 2:
            gates = [SGate()]
        elif value == 3:
            gates = [SGate(), TGate()]
        elif value == 4:
            gates = [ZGate()]
        elif value == 5:
            gates = [SdgGate(), TdgGate()]
        elif value == 6:
            gates = [SdgGate()]
        else:
            gates = [TdgGate()]

        circuit = Circuit(1)
        for gate in gates:
            circuit.append_gate(gate, (0,))
        return CircuitGate(circuit)

    async def run(self, circuit: Circuit, data: PassData) -> None:
        for cycle, op in circuit.operations_with_cycles(reverse=True):
            if not isinstance(op.gate, RZGate):
                continue
            subcircuit = self.check_angle(op.params[0])
            point = (cycle, op.location[0])
            if subcircuit is not None:
                new_op = Operation(subcircuit, op.location)
                circuit.replace(point, new_op)
