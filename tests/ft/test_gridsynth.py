"""This file tests that bqskit.compile outputs are in FaultTolerantGateSet."""
from __future__ import annotations

from math import pi

from random import random

from bqskit.compiler import Compiler
from bqskit.compiler.passdata import PassData
from bqskit.ir import Circuit
from bqskit.ir.gates import RZGate

from bqskit.ft.cliffordt.cliffordtgates import clifford_t_gates
from bqskit.ft.ftpasses.gridsynth import GridSynthPass


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