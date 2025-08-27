"""This module implements a generic FaultTolerantModel class."""
from __future__ import annotations

from typing import Sequence

from bqskit.compiler.registry import register_workflow
from bqskit.ft.cliffordrz.cliffordrzgates import clifford_gates
from bqskit.ft.cliffordrz.cliffordrzgates import rz_gates
from bqskit.ft.cliffordrz.cliffordrzgates import t_gates
from bqskit.ft.cliffordrz.defaultworkflow import build_circuit_workflow
from bqskit.ft.cliffordrz.defaultworkflow import build_statemap_workflow
from bqskit.ft.cliffordrz.defaultworkflow import build_stateprep_workflow
from bqskit.ft.cliffordrz.defaultworkflow import build_unitary_workflow
from bqskit.ft.ftmodel import FaultTolerantModel
from bqskit.ir.gate import Gate


class CliffordRZModel(FaultTolerantModel):

    def __init__(
        self,
        num_qudits: int,
        clifford_gates: Sequence[Gate] = clifford_gates,
        non_clifford_gates: Sequence[Gate] = rz_gates + t_gates,
        radixes: Sequence[int] = [],
    ) -> None:
        """
        Construct a FaultTolerantModel of an error corrected machine.

        Args:
            num_qudits (int): The number of qudits in the machine.

            clifford_gates (Sequence[Gate]): A subset of Clifford gates to
                allow in the model. If not provided, any 1 qubit clifford,
                CNOT, and CZ are allowed.
                (Default: [HGate(), XGate(), YGate(), ZGate(), SGate(),
                SdgGate(), SqrtXGate(), CNOTGate(), CZGate()])

            non_clifford_gates (Sequence[Gate]): A list of non-Clifford
                gates to allow in the model. If RZGates are not desired,
                RZtoCliffordRZSynthesis should be used.
                (Default: [TGate(), TdgGate(), RZGate()])

            radixes (Sequence[int]): The radixes of the qudits. If empty,
                qudits are assumed to be qubits. Currently only qubits
                are supported. (Default: [])

        TODO:
            - Add support for radices >2
        """
        super().__init__(
            num_qudits,
            clifford_gates=clifford_gates,
            non_clifford_gates=non_clifford_gates,
            radixes=radixes,
        )
        for opt_level in [1, 2, 3, 4]:
            register_workflow(
                self,
                build_circuit_workflow(opt_level, decompose_rz=False),
                opt_level,
                'circuit',
            )
            register_workflow(
                self,
                build_unitary_workflow(opt_level, decompose_rz=False),
                opt_level,
                'unitary',
            )
            register_workflow(
                self,
                build_statemap_workflow(opt_level, decompose_rz=False),
                opt_level,
                'statemap',
            )
            register_workflow(
                self,
                build_stateprep_workflow(opt_level, decompose_rz=False),
                opt_level,
                'stateprep',
            )
