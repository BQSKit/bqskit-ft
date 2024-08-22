"""This file tests that bqskit.compile outputs are in FaultTolerantGateSet."""
from __future__ import annotations

from itertools import combinations
from random import choice

from bqskit.compiler import compile
from bqskit.compiler.machine import MachineModel
from bqskit.ir import Circuit
from bqskit.ir import Gate
from bqskit.ir.gates import U3Gate

from bqskit.ft.cliffordt.ftgateset import FaultTolerantGateSet


def simple_circuit(num_qudits: int, gate_set: list[Gate]) -> Circuit:
    circ = Circuit(num_qudits)
    gate = choice(gate_set)
    if gate.num_qudits == 1:
        loc = choice(range(num_qudits))
    else:
        loc = choice(list(combinations(range(num_qudits), 2)))  # type: ignore
    gate_inv = gate.get_inverse()
    circ.append_gate(gate, loc)
    circ.append_gate(gate_inv, loc)
    return circ

class TestCompileDefaults:

    def test_workflow_in_registry(self) -> None:
        input_gateset = [U3Gate()]
        num_qudits = 2
        target = simple_circuit(num_qudits, input_gateset)
        ftgateset = FaultTolerantGateSet()
        machine = MachineModel(num_qudits, gate_set=ftgateset)
        result = compile(target, machine)
        assert all([gate in ftgateset for gate in result.gate_set])