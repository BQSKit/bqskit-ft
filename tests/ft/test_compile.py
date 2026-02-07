"""This file tests that bqskit.compile outputs are in FaultTolerantGateSet."""
from __future__ import annotations

from itertools import combinations
from math import pi
from random import choice
from random import random

from bqskit.compiler.compile import compile
from bqskit.compiler.compiler import Compiler
from bqskit.ft.cliffordrz.cliffordrzgates import clifford_rz_gates
from bqskit.ft.cliffordrz.cliffordrzmodel import CliffordRZModel
from bqskit.ft.cliffordt.cliffordtgates import clifford_t_gates
from bqskit.ft.cliffordt.cliffordtmodel import CliffordTModel
from bqskit.ir import Circuit
from bqskit.ir import Gate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import RXGate
from bqskit.ir.gates import RYGate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import SdgGate
from bqskit.ir.gates import SGate
from bqskit.ir.gates import TdgGate
from bqskit.ir.gates import TGate
from bqskit.ir.gates import U3Gate
from bqskit.ir.gates import ZGate
from bqskit.passes.rules.zxzxz import ZXZXZDecomposition


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

    def test_clifford_is_replaced(self) -> None:
        circuit = Circuit(1)
        circuit.append_gate(HGate(), 0)
        with Compiler() as compiler:
            circuit = compiler.compile(circuit, [ZXZXZDecomposition()])
        assert HGate() not in circuit.gate_set
        model = CliffordRZModel(1)
        result = compile(circuit, model)
        assert HGate() in result.gate_set

    def test_rz_is_replaced_with_z(self) -> None:
        circuit = Circuit(1)
        circuit.append_gate(RZGate(), 0, [pi])
        model = CliffordRZModel(1)
        result = compile(circuit, model)
        assert ZGate() in result.gate_set

    def test_rz_is_replaced_with_s(self) -> None:
        circuit = Circuit(1)
        circuit.append_gate(RZGate(), 0, [pi / 2])
        model = CliffordRZModel(1)
        result = compile(circuit, model)
        assert SGate() in result.gate_set

    def test_rz_is_replaced_with_sdg(self) -> None:
        circuit = Circuit(1)
        circuit.append_gate(RZGate(), 0, [-pi / 2])
        model = CliffordRZModel(1)
        result = compile(circuit, model)
        assert SdgGate() in result.gate_set

    def test_rz_is_replaced_with_t(self) -> None:
        circuit = Circuit(1)
        circuit.append_gate(RZGate(), 0, [pi / 4])
        model = CliffordRZModel(1)
        result = compile(circuit, model)
        assert TGate() in result.gate_set

    def test_rz_is_replaced_with_tdg(self) -> None:
        circuit = Circuit(1)
        circuit.append_gate(RZGate(), 0, [-pi / 4])
        model = CliffordRZModel(1)
        result = compile(circuit, model)
        assert TdgGate() in result.gate_set

    def test_rx_is_replaced_with_rz(self) -> None:
        circuit = Circuit(1)
        circuit.append_gate(RXGate(), 0, [-pi / 8])
        utry_before = circuit.get_unitary()
        model = CliffordRZModel(1)
        result = compile(circuit, model)
        utry_after = result.get_unitary()
        assert RZGate() in result.gate_set
        assert utry_before.get_distance_from(utry_after) < 1e-8

    def test_ry_is_replaced_with_rz(self) -> None:
        circuit = Circuit(1)
        circuit.append_gate(RYGate(), 0, [-pi / 8])
        utry_before = circuit.get_unitary()
        model = CliffordRZModel(1)
        result = compile(circuit, model)
        utry_after = result.get_unitary()
        assert RZGate() in result.gate_set
        assert utry_before.get_distance_from(utry_after) < 1e-8

    def test_num_rzs_doesnt_grow(self) -> None:
        input_gateset = [HGate(), RZGate(), CNOTGate()]
        num_qudits = 3
        target = simple_circuit(num_qudits, input_gateset)  # type: ignore
        while RZGate() not in target.gate_set:
            target = simple_circuit(num_qudits, input_gateset)  # type: ignore
        num_rz_before = target.gate_counts[RZGate()]
        machine = CliffordRZModel(num_qudits)
        result = compile(target, machine)
        num_rz_after = result.gate_counts[RZGate()]
        assert num_rz_after <= num_rz_before
