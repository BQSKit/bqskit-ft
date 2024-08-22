"""This module defines the default FaultTolerantGateSet class."""
from typing import Sequence

from bqskit.compiler.gateset import GateSet

from bqskit.ir.gate import Gate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import XGate
from bqskit.ir.gates import YGate
from bqskit.ir.gates import ZGate
from bqskit.ir.gates import SGate
from bqskit.ir.gates import SdgGate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import SqrtXGate
from bqskit.ir.gates import CZGate
from bqskit.ir.gates import TGate
from bqskit.ir.gates import TdgGate
from bqskit.ir.gates import RZGate


clifford_gates = [
    HGate(),
    XGate(),
    YGate(),
    ZGate(),
    SGate(),
    SdgGate(),
    SqrtXGate(),
    CNOTGate(),
    CZGate(),
]

class FaultTolerantGateSet(GateSet):

    def __init__(
        self,
        clifford_gates: Sequence[Gate] = clifford_gates,
        magic_gates: Sequence[Gate] = [TGate(), TdgGate()],
        parameterized_gates: Sequence[Gate] = [RZGate()],
    ) -> None:
        """
        Construct a FaultTolerantModel of an error corrected machine.

        Args:
            clifford_gates (Sequence[Gate]): A subset of Clifford gates to
                allow in the model. If not provided, any 1 qubit clifford,
                CNOT, and CZ are allowed. (Default: [HGate(), XGate(),
                YGate(), ZGate(), SGate(), SdgGate(), CNOTGate(), CZGate()])

            magic_gates (Sequence[Gate]): A list of non-Clifford gates
                gates which can be implemented with magic state injection.
                These gates must be discrete constant gates.
                (Default: [TGate(), TdgGate()])
            
            parameterized_gates (Sequence[Gate]): A list of non-Clifford
                parameterized gates to allow for synthesis purposes. These
                gates must be handled by a separate synthesis pass before a
                Circuit can be executed. (Defaul: [RZGate()])
        
        TODO:
            - Single qubit synthesis pass for parameterized_gates conversion
        """
        gate_set = clifford_gates + magic_gates + parameterized_gates
        super().__init__(gates=gate_set)