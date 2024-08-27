"""This file tests that bqskit.compile outputs are in FaultTolerantGateSet."""
from __future__ import annotations

import numpy as np

from bqskit.compiler import Compiler
from bqskit.ir import Circuit
from bqskit.ir import Operation
from bqskit.ir.gates import XGate
from bqskit.ir.gates import ZGate
from bqskit.ir.gates import U3Gate
from bqskit.ir.gates import CNOTGate

from bqskit.passes import ForEachBlockPass
from bqskit.passes import GroupSingleQuditGatePass
from bqskit.passes import UnfoldPass

from bqskit.qis.unitary import UnitaryMatrix

from bqskit.ft.replacement import ReplacementRule
from bqskit.ft.replacement import construct_unitary_match_rule


class TestReplacementRules:

    def test_construct_unitary_match(self) -> None:
        x_u = np.array([[0, 1], [1, 0]])
        i_u = np.eye(2)
        op = Operation(XGate(), [0])
        x_match = construct_unitary_match_rule(UnitaryMatrix(x_u))
        i_match = construct_unitary_match_rule(UnitaryMatrix(i_u))
        assert x_match(op)
        assert not i_match(op)
    
    def test_replace_single_qubit(self) -> None:
        x_match = construct_unitary_match_rule(XGate().get_unitary())
        z_match = construct_unitary_match_rule(ZGate().get_unitary())
        x_rule = ReplacementRule(x_match, XGate())
        z_rule = ReplacementRule(z_match, ZGate())
        circuit = Circuit(2)
        circuit.append_gate(U3Gate(), (0), [np.pi, 0, np.pi])
        circuit.append_gate(U3Gate(), (1), [0, np.pi, 0])
        circuit.append_gate(CNOTGate(), (0, 1))
        x_workflow = [
            GroupSingleQuditGatePass(),
            ForEachBlockPass(x_rule),
            UnfoldPass(),
        ]
        z_workflow = [
            GroupSingleQuditGatePass(),
            ForEachBlockPass(z_rule),
            UnfoldPass(),
        ]

        with Compiler() as compiler:
            result = compiler.compile(circuit, x_workflow)

        assert XGate() in result.gate_set
        assert ZGate() not in result.gate_set

        with Compiler() as compiler:
            result = compiler.compile(circuit, z_workflow)

        assert XGate() not in result.gate_set
        assert ZGate() in result.gate_set