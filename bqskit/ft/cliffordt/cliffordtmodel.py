"""This module implements a generic FaultTolerantModel class."""
from __future__ import annotations

from typing import Sequence

from bqskit.compiler.registry import register_workflow
from bqskit.ft.cliffordt.cliffordtgates import clifford_gates
from bqskit.ft.cliffordt.defaultworkflow import build_circuit_workflow
from bqskit.ft.cliffordt.defaultworkflow import build_statemap_workflow
from bqskit.ft.cliffordt.defaultworkflow import build_stateprep_workflow
from bqskit.ft.cliffordt.defaultworkflow import build_unitary_workflow
from bqskit.ft.ftmodel import FaultTolerantModel
from bqskit.ir.gate import Gate
from bqskit.ir.gates.constant.t import TGate
from bqskit.ir.gates.constant.tdg import TdgGate
from bqskit.ir.gates.parameterized.rz import RZGate


class CliffordTModel(FaultTolerantModel):

    def __init__(
        self,
        num_qudits: int,
        algorithmic_error: float = 1e-4,
        clifford_gates: Sequence[Gate] = clifford_gates,
        non_clifford_gates: Sequence[Gate] = [TGate(), TdgGate(), RZGate()],
        radixes: Sequence[int] = [],
        skip_synthesis: bool = False,
        skip_zxzxz: bool = False,
    ) -> None:
        """
        Construct a FaultTolerantModel of an error corrected machine.

        Args:
            num_qudits (int): The number of qudits in the machine.

            algorithmic_error (float): The maximum error allowed across the
                entire circuit. This is a required parameter to decompose
                continous rotations to a discrete gate set.

            clifford_gates (Sequence[Gate]): A subset of Clifford gates to
                allow in the model. If not provided, any 1 qubit clifford,
                CNOT, and CZ are allowed.
                (Default: [HGate(), XGate(), YGate(), ZGate(), SGate(),
                SdgGate(), SqrtXGate(), CNOTGate(), CZGate()])

            non_clifford_gates (Sequence[Gate]): A list of non-Clifford
                gates to allow in the model. If RZGates are not desired,
                RZtoCliffordTSynthesis should be used.
                (Default: [TGate(), TdgGate(), RZGate()])

            radixes (Sequence[int]): The radixes of the qudits. If empty,
                qudits are assumed to be qubits. Currently only qubits
                are supported. (Default: [])

            skip_synthesis (bool): If True, the synthesis passes will to
                re-target the initial gate set will be skipped. You can use this
                if all multi-qubit gates are Clifford. DOES NOT APPLY to
                non-circuit workflows.

            skip_zxzxz (bool): If True, the ZXZXZ decomposition pass will be
                skipped. You can use this if all single-qubit rotation gates are
                Rz gatesand all multi-qubit gates are Clifford. Note that this
                will also skip synthesis. DOES NOT APPLY to non-circuit
                workflows.
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
                build_circuit_workflow(
                    opt_level,
                    algorithmic_error=algorithmic_error,
                    skip_synthesis=skip_synthesis,
                    skip_zxzxz=skip_zxzxz,
                ),
                opt_level,
                'circuit',
            )
            register_workflow(
                self,
                build_unitary_workflow(
                    opt_level,
                    algorithmic_error=algorithmic_error,
                ),
                opt_level,
                'unitary',
            )
            register_workflow(
                self,
                build_statemap_workflow(
                    opt_level,
                    algorithmic_error=algorithmic_error,
                ),
                opt_level,
                'statemap',
            )
            register_workflow(
                self,
                build_stateprep_workflow(
                    opt_level,
                    algorithmic_error=algorithmic_error,
                ),
                opt_level,
                'stateprep',
            )
