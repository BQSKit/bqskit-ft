from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.compile import build_multi_qudit_retarget_workflow
from bqskit.compiler.workflow import Workflow
from bqskit.ft.cliffordt.rounding import RoundToDiscreteZPass
from bqskit.ft.rules.replacement import construct_unitary_match_rule
from bqskit.ft.rules.replacement import ReplacementRule
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.constant.identity import IdentityGate
from bqskit.ir.gates.constant.s import SGate
from bqskit.ir.gates.constant.sdg import SdgGate
from bqskit.ir.gates.constant.sx import SqrtXGate
from bqskit.ir.gates.constant.t import TGate
from bqskit.ir.gates.constant.tdg import TdgGate
from bqskit.ir.gates.constant.x import XGate
from bqskit.ir.gates.constant.y import YGate
from bqskit.ir.gates.constant.z import ZGate
from bqskit.ir.operation import Operation
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.passes.partitioning.quick import QuickPartitioner
from bqskit.passes.partitioning.single import GroupSingleQuditGatePass
from bqskit.passes.processing.scan import ScanningGateRemovalPass
from bqskit.passes.rules.zxzxz import ZXZXZDecomposition
from bqskit.passes.synthesis.qsearch import QSearchSynthesisPass
from bqskit.passes.util.log import LogErrorPass
from bqskit.passes.util.random import SetRandomSeedPass
from bqskit.passes.util.unfold import UnfoldPass
from bqskit.utils.typing import is_real_number


h_repl_rule = construct_unitary_match_rule(HGate().get_unitary())
i_repl_rule = construct_unitary_match_rule(IdentityGate().get_unitary())
x_repl_rule = construct_unitary_match_rule(XGate().get_unitary())
sqrtx_repl_rule = construct_unitary_match_rule(SqrtXGate().get_unitary())
y_repl_rule = construct_unitary_match_rule(YGate().get_unitary())
z_repl_rule = construct_unitary_match_rule(ZGate().get_unitary())
s_repl_rule = construct_unitary_match_rule(SGate().get_unitary())
sdg_repl_rule = construct_unitary_match_rule(SdgGate().get_unitary())
t_repl_rule = construct_unitary_match_rule(TGate().get_unitary())
tdg_repl_rule = construct_unitary_match_rule(TdgGate().get_unitary())


def single_qudit_filter(op: Operation) -> bool:
    return op.num_qudits == 1 and op.num_params > 0


def clifford_replace() -> BasePass:
    return ForEachBlockPass(
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
            ReplacementRule(i_repl_rule, IdentityGate()),
        ],
        collection_filter=single_qudit_filter,
    )

def build_cliffordt_workflow(
    optimization_level: int,
    synthesis_epsilon: float = 1e-8,
    max_synthesis_size: int = 3,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
    circuit_target: bool = False,
    seed: int | None = None,
) -> list[BasePass]:
    """Build a workflow for Clifford+T compilation."""
    passes = [SetRandomSeedPass(seed)] if seed is not None else []
    if circuit_target:
        passes += [UnfoldPass()]
        passes += build_multi_qudit_retarget_workflow(
            optimization_level=optimization_level,
            synthesis_epsilon=synthesis_epsilon,
            max_synthesis_size=max_synthesis_size,
            error_threshold=error_threshold,
            error_sim_size=error_sim_size,
        )
        passes += [UnfoldPass()]
        passes += [QuickPartitioner(block_size=max_synthesis_size)]

    if not circuit_target or optimization_level >= 3:
        passes += build_search_synthesis_workflow(
            optimization_level, synthesis_epsilon,
        )

    passes += [
        GroupSingleQuditGatePass(),
        clifford_replace(),
        UnfoldPass(),
        RoundToDiscreteZPass(synthesis_epsilon),
        QuickPartitioner(2),
        ForEachBlockPass([ScanningGateRemovalPass()]),
        UnfoldPass(),
        GroupSingleQuditGatePass(),
        clifford_replace(),
        UnfoldPass(),
        RoundToDiscreteZPass(synthesis_epsilon),
        UnfoldPass(),
        # Finalizing
        LogErrorPass(),
    ]
    return passes


def build_search_synthesis_workflow(
    optimization_level: int = 1,
    synthesis_epsilon: float = 1e-8,
) -> list[BasePass]:
    """
    Build standard -based synthesis pass for block-level compilation.

    Args:
        optimization_level (int): The optimization level. See :func:`compile`
            for more information.

        synthesis_epsilon (float): The maximum distance between target
            and circuit unitary allowed to declare successful synthesis.
            Set to 0 for exact synthesis. (Default: 1e-8)

    Returns:
        (list[BasePass]): Synthesis passes.

    Raises:
        ValueError: If the optimization level is not 1, 2, 3, or 4.

    Note:
        For larger circuits, this pass may take a very long time to run.
        If unitary synthesis is your ultimate goal, rather than circuit
        compilation, consider designing a custom instantiation-based
        synthesis method or using alternative synthesis techniques
        -- such as QFAST or QPredict -- for large unitaries.
    """
    if optimization_level not in [1, 2, 3, 4]:
        raise ValueError(
            'Invalid optimization level, must be 1, 2, 3, or 4.'
            f' Got {optimization_level}.',
        )

    if not is_real_number(synthesis_epsilon):
        raise TypeError(
            'Expected float for synthesis_epsilon'
            f', got {type(synthesis_epsilon)}.',
        )

    synthesis = QSearchSynthesisPass(success_threshold=synthesis_epsilon)
    group = GroupSingleQuditGatePass()
    foreach = ForEachBlockPass(
        [ZXZXZDecomposition()], collection_filter=single_qudit_filter,
    )

    return [
        synthesis,
        group,
        foreach,
        UnfoldPass(),
    ]


def build_circuit_workflow(
    optimization_level: int = 1,
    synthesis_epsilon: float = 1e-8,
    max_synthesis_size: int = 3,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
    seed: int | None = None,
) -> Workflow:
    """Build standard workflow for circuit compilation."""
    workflow = build_cliffordt_workflow(
        optimization_level,
        synthesis_epsilon,
        max_synthesis_size,
        error_threshold,
        error_sim_size,
        circuit_target=True,
        seed=seed,
    )
    return Workflow(
        workflow, name='Off-the-Shelf Clifford+T Circuit Compilation',
    )


def build_unitary_workflow(
    optimization_level: int = 1,
    synthesis_epsilon: float = 1e-8,
    max_synthesis_size: int = 3,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
    seed: int | None = None,
) -> Workflow:
    """Build standard workflow for circuit compilation."""
    workflow = build_cliffordt_workflow(
        optimization_level=optimization_level,
        synthesis_epsilon=synthesis_epsilon,
        max_synthesis_size=max_synthesis_size,
        error_threshold=error_threshold,
        error_sim_size=error_sim_size,
        circuit_target=False,
        seed=seed,
    )
    return Workflow(
        workflow, name='Off-the-Shelf Clifford+T Unitary Compilation',
    )


def build_statemap_workflow(
    optimization_level: int = 1,
    synthesis_epsilon: float = 1e-8,
    max_synthesis_size: int = 3,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
    seed: int | None = None,
) -> Workflow:
    """Build standard workflow for circuit compilation."""
    workflow = build_cliffordt_workflow(
        optimization_level,
        synthesis_epsilon,
        max_synthesis_size,
        error_threshold,
        error_sim_size,
        circuit_target=False,
        seed=seed,
    )
    return Workflow(
        workflow, name='Off-the-Shelf Clifford+T StateSystem Compilation',
    )


def build_stateprep_workflow(
    optimization_level: int = 1,
    synthesis_epsilon: float = 1e-8,
    max_synthesis_size: int = 3,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
    seed: int | None = None,
) -> Workflow:
    """Build standard workflow for circuit compilation."""
    workflow = build_cliffordt_workflow(
        optimization_level,
        synthesis_epsilon,
        max_synthesis_size,
        error_threshold,
        error_sim_size,
        circuit_target=False,
        seed=seed,
    )
    return Workflow(
        workflow, name='Off-the-Shelf Clifford+T StateVector Compilation',
    )
