"""This module implements a generic FaultTolerantModel class."""
from typing import Sequence

from bqskit.compiler.machine import MachineModel

from bqskit.ir.gate import Gate


class FaultTolerantModel(MachineModel):

    def __init__(
        self,
        num_qudits: int,
        clifford_gates: Sequence[Gate],
        non_clifford_gates: Sequence[Gate],
        radixes: Sequence[int] = [],
    ) -> None:
        """
        Construct a FaultTolerantModel of an error corrected machine.

        Args:
            num_qudits (int): The number of qudits in the machine.

            clifford_gates (Sequence[Gate]): A subset of Clifford gates to
                allow in the model.

            non_clifford_gates (Sequence[Gate]): A list of non-Clifford
                gates to allow in the model.
            
            radixes (Sequence[int]): The radixes of the qudits. If empty,
                qudits are assumed to be qubits. (Default: [])
        """
        gate_set = clifford_gates + non_clifford_gates
        super().__init__(num_qudits, gate_set=gate_set, radixes=radixes)