from __future__ import annotations

from math import ceil, log10

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.compile import build_multi_qudit_retarget_workflow
from bqskit.compiler.passdata import PassData
from bqskit.compiler.workflow import Workflow
from bqskit.ft.ftpasses.dpf import DPFPass
from bqskit.ft.ftpasses.convert_frz import ConvertFractionalRZPass
from bqskit.ft.ftpasses.gridsynth import GridSynthPass
from bqskit.ft.ftpasses.rounding import RoundToDiscreteZPass
from bqskit.ft.rules.replacement import construct_unitary_match_rule
from bqskit.ft.rules.replacement import ReplacementRule
from bqskit.ft.rules.xytoz import XYtoZRotation
from bqskit.ir.circuit import Circuit
from bqskit.ir.circuit import CircuitGate
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
from bqskit.ir.gates.parameterized.rx import RXGate
from bqskit.ir.gates.parameterized.ry import RYGate
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.ir.operation import Operation
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.passes.partitioning.quick import QuickPartitioner
from bqskit.passes.partitioning.single import GroupSingleQuditGatePass
from bqskit.passes.rules.zxzxz import ZXZXZDecomposition
from bqskit.passes.synthesis.qsearch import QSearchSynthesisPass
from bqskit.passes.util.extend import ExtendBlockSizePass
from bqskit.passes.util.log import LogErrorPass
from bqskit.passes.util.random import SetRandomSeedPass
from bqskit.passes.util.unfold import UnfoldPass
from bqskit.passes.util.update import UpdateDataPass
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


def single_qudit_u2_or_u3(op: Operation) -> bool:
    return op.num_qudits == 1 and op.num_params > 1


def single_qudit_rx_or_ry(op: Operation) -> bool:
    is_ry = isinstance(op.gate, RYGate)
    is_rx = isinstance(op.gate, RXGate)
    return op.num_qudits == 1 and (is_ry or is_rx)


def rz_gate_filter(op: Operation) -> bool:
    return isinstance(op.gate, RZGate)


def clifford_replace() -> BasePass:
    return ForEachBlockPass(
        [
            ReplacementRule(h_repl_rule, HGate()),  # type: ignore
            ReplacementRule(x_repl_rule, XGate()),  # type: ignore
            ReplacementRule(sqrtx_repl_rule, SqrtXGate()),  # type: ignore
            ReplacementRule(y_repl_rule, YGate()),  # type: ignore
            ReplacementRule(z_repl_rule, ZGate()),  # type: ignore
            ReplacementRule(s_repl_rule, SGate()),  # type: ignore
            ReplacementRule(sdg_repl_rule, SdgGate()),  # type: ignore
            ReplacementRule(t_repl_rule, TGate()),  # type: ignore
            ReplacementRule(tdg_repl_rule, TdgGate()),  # type: ignore
            ReplacementRule(i_repl_rule, None),  # type: ignore
        ],
        collection_filter=single_qudit_filter,
    )


class AssignErrors(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        num_blocks = 0
        for op in circuit.operations():
            if isinstance(op.gate, CircuitGate):
                num_blocks += 1
        algorithmic_error = data['algorithmic_error']
        pass_down_label = (
            ForEachBlockPass.pass_down_key_prefix
            + 'algorithmic_error'
        )
        data[pass_down_label] = algorithmic_error / num_blocks


class ClearBlockError(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        pass_down_label = (
            ForEachBlockPass.pass_down_key_prefix
            + 'algorithmic_error'
        )
        data.pop(pass_down_label, None)


def build_error_aware_partitioning_workflow(
    core_workflow: list[BasePass],
    block_size: int = 3,
    replace_filter: str = 'always',
) -> list[BasePass]:
    """Build a partitioning workflow passes down the error per circuit """
    # Partition Circuit
    pass_list = [QuickPartitioner(block_size), ExtendBlockSizePass()]

    # Assign the error to each block
    pass_list += [AssignErrors()]

    # Now generate the ForEachBlockPass for the core workflow
    pass_list += [
        ForEachBlockPass(
            core_workflow,
            replace_filter=replace_filter,
        ),
    ]

    pass_list += [UnfoldPass(), ClearBlockError()]
    return Workflow(pass_list, name='Partitioning')


def build_cliffordt_workflow(
    optimization_level: int = 1,
    algorithmic_error: float = 1e-3,
    max_synthesis_size: int = 3,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
    circuit_target: bool = False,
    decompose_rz: bool = True,
    seed: int | None = None,
    skip_synthesis: bool = False,
    skip_zxzxz: bool = False,
    use_ccz: bool = False, # For now just skip synthesis
    synthesis_epsilon: float = 1e-8,
) -> list[BasePass]:
    """Build a workflow for Clifford+T compilation."""
    passes = [SetRandomSeedPass(seed)] if seed is not None else []
    passes += [UpdateDataPass('algorithmic_error', algorithmic_error)]
    if circuit_target:
        passes += [UnfoldPass()]
        if not skip_synthesis and not use_ccz:
            passes += build_multi_qudit_retarget_workflow(
                optimization_level=optimization_level,
                synthesis_epsilon=synthesis_epsilon,
                max_synthesis_size=max_synthesis_size,
                error_threshold=error_threshold,
                error_sim_size=error_sim_size,
            )

    if not circuit_target:
        assert skip_synthesis is False
        passes += build_search_synthesis_workflow(
            optimization_level, synthesis_epsilon,
        )

    # Build Core Workflow for Clifford + Single Qubit Rotations -> Clifford + T
    core_workflow = []

    zxzxz = ForEachBlockPass(
        [ZXZXZDecomposition()], collection_filter=single_qudit_u2_or_u3,
    )
    xytoz = ForEachBlockPass(
        [XYtoZRotation()], collection_filter=single_qudit_rx_or_ry,
    )

    core_workflow += [
        # --------------------------------------------------
        # Replace single qudit Cliffords where possible.
        # --------------------------------------------------
        GroupSingleQuditGatePass(),
        clifford_replace(),
        # --------------------------------------------------
        # Convert RX and RY gates to RZ gates.
        # --------------------------------------------------
        UnfoldPass(),
        xytoz,
        # --------------------------------------------------
        # Replace Z, S, Sdg, T, and Tdg gates when possible.
        # --------------------------------------------------
        RoundToDiscreteZPass(),
    ]

    # --------------------------------------------------
    # Convert to Clifford + Rz gate set.
    # --------------------------------------------------
    if not skip_zxzxz:
        core_workflow += [
            GroupSingleQuditGatePass(),
            zxzxz,
            clifford_replace(),
            UnfoldPass(),
            RoundToDiscreteZPass(),
            UnfoldPass(),
        ]

    # --------------------------------------------------
    # Now, use Dyadic Phase Fixing to remove more Rzs
    # --------------------------------------------------
    core_workflow += [
        DPFPass(),
        ConvertFractionalRZPass(),
    ]

    # Put into partitioned workflow
    passes += build_error_aware_partitioning_workflow(
        core_workflow,
        block_size=max_synthesis_size,
        replace_filter='always',
    )

    # Decompose RZ gates into Clifford+T
    if decompose_rz:
        passes += [GridSynthPass()]

    passes += [LogErrorPass()]
    return passes  # type: ignore


def build_search_synthesis_workflow(
    optimization_level: int = 1,
    synthesis_epsilon: float = 1e-8,
    decompose_rz: bool = False,
) -> list[BasePass]:
    """
    Build standard -based synthesis pass for block-level compilation.

    Args:
        optimization_level (int): The optimization level. See :func:`compile`
            for more information.

        synthesis_epsilon (float): The maximum distance between target
            and circuit unitary allowed to declare successful synthesis.
            Set to 0 for exact synthesis. (Default: 1e-8)

        decompose_rz (bool): Whether to decompose RZ gates into Clifford+T
            (Default: False).

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
    foreach = ForEachBlockPass(
        [ZXZXZDecomposition()], collection_filter=single_qudit_u2_or_u3,
    )

    passes = [synthesis, foreach, UnfoldPass()]

    if decompose_rz:
        passes += [GridSynthPass(synthesis_epsilon)]

    return passes


def build_circuit_workflow(
    optimization_level: int = 1,
    algorithmic_error: float = 1e-3,
    max_synthesis_size: int = 3,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
    decompose_rz: bool = True,
    seed: int | None = None,
    skip_synthesis: bool = False,
    skip_zxzxz: bool = False,
    synthesis_epsilon: float = 1e-8,
    use_ccz: bool = False,
) -> Workflow:
    """Build standard workflow for circuit compilation."""
    workflow = build_cliffordt_workflow(
        optimization_level,
        algorithmic_error,
        max_synthesis_size,
        error_threshold,
        error_sim_size,
        circuit_target=True,
        decompose_rz=decompose_rz,
        seed=seed,
        skip_synthesis=skip_synthesis,
        skip_zxzxz=skip_zxzxz,
        synthesis_epsilon=synthesis_epsilon,
        use_ccz=use_ccz,
    )
    return Workflow(
        workflow, name='Off-the-Shelf Clifford+T Circuit Compilation',
    )


def build_unitary_workflow(
    optimization_level: int = 1,
    algorithmic_error: float = 1e-3,
    synthesis_epsilon: float = 1e-8,
    max_synthesis_size: int = 3,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
    seed: int | None = None,
    decompose_rz: bool = True,
) -> Workflow:
    """Build standard workflow for circuit compilation."""
    workflow = build_cliffordt_workflow(
        optimization_level=optimization_level,
        algorithmic_error=algorithmic_error,
        synthesis_epsilon=synthesis_epsilon,
        max_synthesis_size=max_synthesis_size,
        error_threshold=error_threshold,
        error_sim_size=error_sim_size,
        circuit_target=False,
        decompose_rz=decompose_rz,
        seed=seed,
    )
    return Workflow(
        workflow, name='Off-the-Shelf Clifford+T Unitary Compilation',
    )


def build_statemap_workflow(
    optimization_level: int = 1,
    algorithmic_error: float = 1e-3,
    max_synthesis_size: int = 3,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
    decompose_rz: bool = True,
    seed: int | None = None,
    synthesis_epsilon: float = 1e-8,
) -> Workflow:
    """Build standard workflow for circuit compilation."""
    workflow = build_cliffordt_workflow(
        optimization_level,
        algorithmic_error,
        max_synthesis_size,
        error_threshold,
        error_sim_size,
        circuit_target=False,
        decompose_rz=decompose_rz,
        seed=seed,
        synthesis_epsilon=synthesis_epsilon,
    )
    return Workflow(
        workflow, name='Off-the-Shelf Clifford+T StateSystem Compilation',
    )


def build_stateprep_workflow(
    optimization_level: int = 1,
    algorithmic_error: float = 1e-3,
    max_synthesis_size: int = 3,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
    decompose_rz: bool = True,
    seed: int | None = None,
    synthesis_epsilon: float = 1e-8,
) -> Workflow:
    """Build standard workflow for circuit compilation."""
    workflow = build_cliffordt_workflow(
        optimization_level,
        algorithmic_error,
        max_synthesis_size,
        error_threshold,
        error_sim_size,
        circuit_target=False,
        decompose_rz=decompose_rz,
        seed=seed,
        synthesis_epsilon=synthesis_epsilon,
    )
    return Workflow(
        workflow, name='Off-the-Shelf Clifford+T StateVector Compilation',
    )
