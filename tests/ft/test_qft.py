"""This file tests that bqskit.compile outputs are in FaultTolerantGateSet."""
from __future__ import annotations

from numpy import array
from numpy import exp
from numpy import fromfunction
from numpy import isclose
from numpy import ndarray
from numpy import pi
from numpy import sqrt

from bqskit.ft.gadgets.qft import QFTGadget
from bqskit.ir.circuit import Circuit


def qft_unitary(num_qubits: int) -> ndarray:
    n = 2 ** num_qubits
    root = exp(2j * pi / n)
    return array(fromfunction(lambda x, y: root**(x * y), (n, n)) / sqrt(n))


class TestCompileDefaults:
    def test_qft_1(self) -> None:
        num_qubits = 1
        circuit = Circuit(num_qubits)
        qft = QFTGadget.generate(num_qubits)
        circuit.append_circuit(qft, [_ for _ in range(num_qubits)])
        circuit.unfold_all()
        print('=' * 80)
        u_qft = qft_unitary(num_qubits)
        dist = circuit.get_unitary().get_distance_from(u_qft)
        assert isclose(dist, 0.0)

    def test_qft_2(self) -> None:
        num_qubits = 2
        circuit = Circuit(num_qubits)
        qft = QFTGadget.generate(num_qubits)
        circuit.append_circuit(qft, [_ for _ in range(num_qubits)])
        circuit.unfold_all()
        print('=' * 80)
        u_qft = qft_unitary(num_qubits)
        dist = circuit.get_unitary().get_distance_from(u_qft)
        assert isclose(dist, 0.0)

    def test_qft_3(self) -> None:
        num_qubits = 3
        circuit = Circuit(num_qubits)
        qft = QFTGadget.generate(num_qubits)
        circuit.append_circuit(qft, [_ for _ in range(num_qubits)])
        circuit.unfold_all()
        print('=' * 80)
        u_qft = qft_unitary(num_qubits)
        dist = circuit.get_unitary().get_distance_from(u_qft)
        assert isclose(dist, 0.0)

    def test_qft_4(self) -> None:
        num_qubits = 4
        circuit = Circuit(num_qubits)
        qft = QFTGadget.generate(num_qubits)
        circuit.append_circuit(qft, [_ for _ in range(num_qubits)])
        circuit.unfold_all()
        print('=' * 80)
        u_qft = qft_unitary(num_qubits)
        dist = circuit.get_unitary().get_distance_from(u_qft)
        assert isclose(dist, 0.0)

    def test_qft_8(self) -> None:
        num_qubits = 8
        circuit = Circuit(num_qubits)
        qft = QFTGadget.generate(num_qubits)
        circuit.append_circuit(qft, [_ for _ in range(num_qubits)])
        circuit.unfold_all()
        print('=' * 80)
        u_qft = qft_unitary(num_qubits)
        dist = circuit.get_unitary().get_distance_from(u_qft)
        assert isclose(dist, 0.0)
