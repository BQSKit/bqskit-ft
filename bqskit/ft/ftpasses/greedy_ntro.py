"""Modified NTRO to work well for PKAC decomposition"""
from __future__ import annotations

import numpy as np
from pygridsynth.gridsynth import gridsynth_gates

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.compiler.gateset import GateSet
from bqskit.ir.circuit import Circuit, CircuitPoint
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.t import TGate
from bqskit.ir.gates.constant.tdg import TdgGate
from bqskit.ir.gates.parameterized.rz import RZGate

from bqskit.ir.opt.cost.functions import HilbertSchmidtCostGenerator
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from ntro.tcount import MatrixDistanceCostGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator

from bqskit.runtime import get_runtime

from bqskit.ft.cliffordrz.cliffordrzgates import clifford_rz_gates
from bqskit.ft.gates.fractional_rz import FractionalRZGate


class GreedyNTROPass(BasePass):
    def __init__(
        self,
        max_k: int,
        success_threshold: float = 1e-8,
        cost: CostFunctionGenerator = MatrixDistanceCostGenerator(),
    ) -> None:
        self.k = max_k
        self.success_threshold = success_threshold
        self.base_period = 2 * np.pi / (2 ** self.k)
        self.cost = cost
        self.instantiate_options = {
            'cost_fn_gen': self.cost,
            'multistarts': 4,
            'ftol': 5e-16,
            'gtol': 1e-15,
            'diff_tol_r': 5e-5,
            'max_iters': 50000,
            'min_iters': 200,
        }


    def fix_closest_angle(self, circuit: Circuit, n: int) -> Circuit:
        '''
        Finds the (n + 1)th RZ angle that is closest to a multiple of
        2pi/(2 ** self.k). Then, replaces that RZ with the corresponding 
        FractionalRZ gate.

        If n >= number of RZ Gates in the circuit, return None.
        '''
        num_rzs = circuit.count(RZGate())

        if n >= num_rzs:
            return None
        
        # Otherwise look at params
        rz_params = np.array(circuit.params)

        # Get distances from closest multiple of base period
        mod_params = np.round(rz_params / self.base_period)
        rounded_params = np.round(mod_params) * self.base_period
        

        diffs = np.abs(rz_params - rounded_params)

        # Get nth smallest diff
        ind = np.argsort(diffs)[n]
        num = mod_params[ind]
        # Make num between 0 and 2**k - 1
        num = num % (2 ** self.k)

        # Now replace RZ gate at ind with Fractional
        rz_ind = 0
        new_circ = circuit.copy()
        for cycle, op in new_circ.operations_with_cycles():
            if isinstance(op.gate, RZGate):
                if rz_ind == ind:
                    new_circ.replace_gate(
                        CircuitPoint(cycle, op.location[0]),
                        FractionalRZGate(num, self.k),
                        op.location
                    )
                    return new_circ
                else:
                    rz_ind += 1
                
        assert False, f"Was not able to replace gate {circuit.gate_counts}, {ind}, {n}, {diffs}"


    async def greedy_search(self, circuit: Circuit, 
                            target: UnitaryMatrix) -> Circuit:

        best_circuit = circuit.copy()

        candidate = self.fix_closest_angle(circuit, 0)
        rz_ind = 0
        fallback = circuit

        while candidate is not None:
            inst_circ = candidate.instantiate(
                target=target,
                instantiation_options=self.instantiate_options
            )
            cost = self.cost.calc_cost(inst_circ, target=target)
            if cost < self.success_threshold:
                if inst_circ.num_params < best_circuit.num_params:
                    best_circuit = inst_circ

                fallback = candidate
                rz_ind = 0
                candidate = self.fix_closest_angle(candidate, rz_ind)
            else:
                rz_ind += 1
                candidate = self.fix_closest_angle(fallback, rz_ind)
                    
        return best_circuit

    def choose_best_circuit(self, circuits: list[Circuit], target: UnitaryMatrix) -> Circuit:
        '''
        Give each circuit a score in terms of resources:

        In general, we will do the following cost function:

        Ts = 1
        RZs = -1 * log10(self.success_threshold) * 10

        Fractional RZs =  4*k -4 

        (TODO: Consider CNOT cost as well)
        '''

        def cost_fn(circ: Circuit) -> float:
            num_t = circ.count(TGate()) + circ.count(TdgGate())
            num_rz = circ.count(RZGate())
            frac_z_cost = 0
            for op in circ.operations():
                if isinstance(op.gate, FractionalRZGate):
                    if op.gate.k == 3:
                        frac_z_cost += 1
                    elif op.gate.k == 4:
                        frac_z_cost += 4
                    else:
                        frac_z_cost += 4 * op.gate.k - 4

            rz_cost_per_gate = np.ceil(-1 * np.log10(self.success_threshold)) * 10
            return num_t + num_rz * rz_cost_per_gate + frac_z_cost

        best_circuit = min(circuits, key=cost_fn)
        return best_circuit       
        
    async def run(self, circuit: Circuit, data: PassData) -> None:
        # circuit_gates = circuit.gate_set
        # # Assert all gates in circuit_gates are in clifford_rz_gates
        # cliff_rz_gates = GateSet(clifford_rz_gates)
        # assert circuit_gates.issubset(cliff_rz_gates)
        prev_circs: list[Circuit] = data.get('prev_ntro_circs', [circuit.copy()])

        # orig_rzs = circuit.count(RZGate())
        # if orig_rzs == 0:
        #     # No need to search
        #     return

        # rz_counts = [circ.count(RZGate()) for circ in prev_circs]

        # print("RZ Counts: ", rz_counts, flush=True)
        
        new_circs = await get_runtime().map(self.greedy_search,
                                            prev_circs, 
                                            target=data.target)
        
        best_circ = self.choose_best_circuit(new_circs, data.target)

        prev_circs.extend(new_circs)
        data['prev_ntro_circs'] = prev_circs
        circuit.become(best_circ)


class ReplaceFractionalRZWithT(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, FractionalRZGate) and op.gate.k <= 3:
                circuit.replace_with_circuit(
                    CircuitPoint(cycle, op.location[0]),
                    FractionalRZGate.get_circuit(op.gate.numerator, op.gate.k),
                    as_circuit_gate=True
                )
        circuit.unfold_all()