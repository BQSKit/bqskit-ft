from __future__ import annotations
from itertools import cycle

from numpy import pi
from numpy import round
from numpy import random

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.cz import CZGate
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.measure import MeasurementPlaceholder
from bqskit.ir.point import CircuitPoint

from bqskit.ir.operation import Operation
from bqskit.ir.gates import CircuitGate

from bqskit.ft.gates.logical_and import LogicalAndDgGate


class PKACtoGatesPass(BasePass):
    '''
    Pass that converts all GidneyAdders, QFTs, and LogicalAnds to gates 
    placeable in tilers. Importantly, this pass does *not* keep the unitary
    of the circuit the same, since we replace LogicalAndInverses with an
    H gate and control Z (which is applied 50% of the time). 
    This is because we want to be able to test the PKAC mapping.
    '''
    async def run(self, circuit: Circuit, data: PassData) -> None:
        # First step, unfold all the gadgets in the circuit
        # This will unfold all the GidneyAdders
        circuit.unfold_all()

        base_log_and_circ = Circuit(3)
        base_log_and_circ.append_gate(HGate(), [2])
        base_log_and_circ.append_gate(MeasurementPlaceholder([("a", 1)], {2: ("a", 0)}), [2])

        # Now, we should replace all LogicalAndDgs with the measure and fixup
        pts = []
        new_circuit_gates = []
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, LogicalAndDgGate):
                # Replace with H, measurement and control Z
                new_circ = base_log_and_circ.copy()
                # Apply a CZ gate with 50% probability
                if random.rand() < 0.5:
                    new_circ.append_gate(CZGate(), [0, 1])

                pt = CircuitPoint(cycle, op.location[0])
                new_circ_gate = CircuitGate(new_circ)
                new_circ_op = Operation(new_circ_gate, op.location)
                pts.append(pt)
                new_circuit_gates.append(new_circ_op)
                
        circuit.batch_replace(pts, new_circuit_gates)

        # Unfold all LogicalAndDgs
        circuit.unfold_all()
