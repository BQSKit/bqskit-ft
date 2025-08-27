"""This file tests that bqskit.compile outputs are in FaultTolerantGateSet."""
from __future__ import annotations

from math import pi
from random import random

from bqskit.compiler import Compiler
from bqskit.ft.cliffordt.cliffordtgates import clifford_t_gates
from bqskit.ft.ftpasses.gridsynth import GridSynthPass
from bqskit.ft.rules.isolate_rz import IsolateRZGatePass
from bqskit.ir import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import U3Gate
from bqskit.passes import ForEachBlockPass
from bqskit.passes import UnfoldPass


class TestGridSynthPass:

    num_trials = 100

    def test_gridsynth(self) -> None:
        for _ in range(self.num_trials):

            circuit = Circuit(1)
            theta = random() * 2 * pi
            circuit.append_gate(RZGate(), [0], [theta])

            gridsynth = GridSynthPass(precision=20)
            old_circuit = circuit.copy()

            with Compiler() as compiler:
                new_circuit = compiler.compile(circuit, [gridsynth])

            for op in new_circuit:
                assert op.gate in clifford_t_gates

            old_utry = old_circuit.get_unitary()
            new_utry = new_circuit.get_unitary()

            assert old_utry.get_distance_from(new_utry) < 1e-8

    def test_gridsynth_in_circuit(self) -> None:
        circuit = Circuit(2)
        theta0 = random() * 2 * pi
        theta1 = random() * 2 * pi
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(RZGate(), [0], [theta0])
        circuit.append_gate(RZGate(), [1], [theta1])

        old_circuit = circuit.copy()

        passes = [
            IsolateRZGatePass(),
            ForEachBlockPass([GridSynthPass(precision=20)]),
            UnfoldPass(),
        ]

        with Compiler() as compiler:
            new_circuit = compiler.compile(circuit, passes)

        for op in new_circuit:
            assert op.gate in clifford_t_gates or op.gate == U3Gate()

        old_utry = old_circuit.get_unitary()
        new_utry = new_circuit.get_unitary()

        assert old_utry.get_distance_from(new_utry) < 1e-8
