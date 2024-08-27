from bqskit.compiler.registry import register_workflow
from bqskit.compiler.gateset import GateSet

from bqskit.ir.gates import HGate
from bqskit.ir.gates import XGate
from bqskit.ir.gates import SqrtXGate
from bqskit.ir.gates import YGate
from bqskit.ir.gates import ZGate
from bqskit.ir.gates import SGate
from bqskit.ir.gates import SdgGate
from bqskit.ir.gates import TGate
from bqskit.ir.gates import TdgGate
from bqskit.ir.operation import Operation

from bqskit.passes import GroupSingleQuditGatePass
from bqskit.passes import ScanningGateRemovalPass
from bqskit.passes import ForEachBlockPass
from bqskit.passes import ZXZXZDecomposition
from bqskit.passes import UnfoldPass

from .cliffordt.ftgateset import FaultTolerantGateSet
from bqskit.ft.replacement import construct_unitary_match_rule
from bqskit.ft.replacement import ReplacementRule


def single_qudit_filter(op: Operation) -> bool:
    return op.num_qudits == 1 and op.num_params > 0

h_repl_rule = construct_unitary_match_rule(HGate().get_unitary())
x_repl_rule = construct_unitary_match_rule(XGate().get_unitary())
sqrtx_repl_rule = construct_unitary_match_rule(SqrtXGate().get_unitary())
y_repl_rule = construct_unitary_match_rule(YGate().get_unitary())
z_repl_rule = construct_unitary_match_rule(ZGate().get_unitary())
s_repl_rule = construct_unitary_match_rule(SGate().get_unitary())
sdg_repl_rule = construct_unitary_match_rule(SdgGate().get_unitary())
t_repl_rule = construct_unitary_match_rule(TGate().get_unitary())
tdg_repl_rule = construct_unitary_match_rule(TdgGate().get_unitary())


clifford_replace = ForEachBlockPass(
    [
        ReplacementRule(h_repl_rule, HGate()),
        ReplacementRule(x_repl_rule, XGate()),
        ReplacementRule(sqrtx_repl_rule, SqrtXGate()),
        ReplacementRule(y_repl_rule, YGate()),
        ReplacementRule(z_repl_rule, ZGate()),
        ReplacementRule(s_repl_rule, SGate()),
        ReplacementRule(sdg_repl_rule, SdgGate()),
        ReplacementRule(t_repl_rule, TGate()),
        ReplacementRule(tdg_repl_rule, TdgGate()),
    ],
    collection_filter=single_qudit_filter,
)


level_1_workflow = [
    GroupSingleQuditGatePass(),
    clifford_replace,
    ForEachBlockPass(
        [ScanningGateRemovalPass(), ZXZXZDecomposition()],
        collection_filter=single_qudit_filter,
    ),
    UnfoldPass(),
]
register_workflow(FaultTolerantGateSet(), level_1_workflow, 1)

# level_2_passes = []
# level_3_passes = []