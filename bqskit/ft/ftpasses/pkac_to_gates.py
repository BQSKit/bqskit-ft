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
from bqskit.ir.point import CircuitPoint

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

        # Now, we should replace all LogicalAndDgs with the measure and fixup
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, LogicalAndDgGate):
                # Replace with H and control Z
                new_circ = Circuit(3)
                new_circ.append_gate(HGate(), [2])
                # Apply a CZ gate with 50% probability
                if random.rand() < 0.5:
                    new_circ.append_gate(CZGate(), [0, 1])

                pt = CircuitPoint(cycle, op.location[0])
                circuit.replace_with_circuit(pt, new_circ, as_circuit_gate=True)

        # Unfold all LogicalAndDgs
        circuit.unfold_all()
