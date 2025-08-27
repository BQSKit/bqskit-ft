"""This file tests that bqskit.compile outputs are in FaultTolerantGateSet."""
from __future__ import annotations

from itertools import combinations
from math import pi
from random import choice
from random import random

from bqskit.compiler.compile import compile
from bqskit.ft.cliffordrz.cliffordrzgates import clifford_rz_gates
from bqskit.ft.cliffordrz.cliffordrzmodel import CliffordRZModel
from bqskit.ft.cliffordt.cliffordtgates import clifford_t_gates
from bqskit.ft.cliffordt.cliffordtmodel import CliffordTModel
from bqskit.ir import Circuit
from bqskit.ir import Gate
from bqskit.ir.gates import U3Gate


def trivial_circuit(num_qudits: int, gate_set: list[Gate]) -> Circuit:
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


def simple_circuit(num_qudits: int, gate_set: list[Gate]) -> Circuit:
    circ = Circuit(num_qudits)
    gate = choice(gate_set)
    if gate.num_qudits == 1:
        loc = choice(range(num_qudits))
    else:
        loc = choice(list(combinations(range(num_qudits), 2)))  # type: ignore
    circ.append_gate(gate, loc)
    params = [2 * pi * random() for _ in range(circ.num_params)]
    circ.set_params(params)
    return circ


class TestCompileDefaults:

    def test_rz_workflow_in_registry(self) -> None:
        input_gateset = [U3Gate()]
        num_qudits = 2
        target = simple_circuit(num_qudits, input_gateset)  # type: ignore
        ftgateset = clifford_rz_gates
        machine = CliffordRZModel(num_qudits)
        result = compile(target, machine)
        assert all([gate in ftgateset for gate in result.gate_set])

    def test_trivial_t_workflow_in_registry(self) -> None:
        input_gateset = [U3Gate()]
        num_qudits = 2
        target = simple_circuit(num_qudits, input_gateset)  # type: ignore
        ftgateset = clifford_t_gates
        machine = CliffordTModel(num_qudits)
        result = compile(target, machine)
        assert all([gate in ftgateset for gate in result.gate_set])

    def test_t_workflow_in_registry(self) -> None:
        input_gateset = [U3Gate()]
        num_qudits = 2
        target = simple_circuit(num_qudits, input_gateset)  # type: ignore
        ftgateset = clifford_t_gates
        machine = CliffordTModel(num_qudits)
        result = compile(target, machine)
        assert len(result.gate_set) > 0
        assert all([gate in ftgateset for gate in result.gate_set])
