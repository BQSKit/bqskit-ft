from bqskit.compiler.registry import register_workflow
from bqskit.compiler.gateset import GateSet

from bqskit.ir.operation import Operation

from bqskit.passes import GroupSingleQuditGatePass
from bqskit.passes import ScanningGateRemovalPass
from bqskit.passes import ForEachBlockPass
from bqskit.passes import ZXZXZDecomposition
from bqskit.passes import UnfoldPass

from .cliffordt.ftgateset import FaultTolerantGateSet


def single_qudit_filter(op: Operation) -> bool:
    return op.num_qudits == 1

level_1_workflow = [
    GroupSingleQuditGatePass(),
    ForEachBlockPass(
        [ScanningGateRemovalPass(), ZXZXZDecomposition()],
        collection_filter=single_qudit_filter,
    ),
    UnfoldPass(),
]
register_workflow(FaultTolerantGateSet(), level_1_workflow, 1)

# level_2_passes = []
# level_3_passes = []