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

from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ft.ftpasses.greedy_ntro import MatrixDistanceCostGenerator

from bqskit.runtime import get_runtime

from bqskit.ft.cliffordrz.cliffordrzgates import clifford_rz_gates
from bqskit.ft.gates.fractional_rz import FractionalRZGate


class ExhaustiveNTROPass(BasePass):
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


    def fix_all_angles(self, circuit: Circuit) -> list[Circuit]:
        '''
        Finds the (n + 1)th RZ angle that is closest to a multiple of
        2pi/(2 ** self.k). Then, replaces that RZ with the corresponding 
        FractionalRZ gate.

        If n >= number of RZ Gates in the circuit, return None.
        '''
        num_rzs = circuit.count(RZGate())

        # Get all binary strings of length num_rzs
        all_strings = [bin(i)[2:].zfill(num_rzs) for i in range(2 ** num_rzs)]
        # 0 pad each string to length num_rzs
        all_strings = [s.zfill(num_rzs) for s in all_strings]

        all_circs = []
        for s in all_strings:
            # If s[i] == 1, replace ith RZ with FractionalRZ
            new_circ = circuit.copy()
            rz_ind = 0
            for cycle, op in new_circ.operations_with_cycles():
                if isinstance(op.gate, RZGate):
                    if s[rz_ind] == '1':
                        # Get angle of RZ gate
                        angle = op.params[0]
                        num = np.round(angle / self.base_period) % (2 ** self.k)
                        new_circ.replace_gate(
                            CircuitPoint(cycle, op.location[0]),
                            FractionalRZGate(num, self.k),
                            op.location
                        )
                    rz_ind += 1
            all_circs.append(new_circ)

        return all_circs


    async def exhaustive_search(self, circuit: Circuit, 
                                target: UnitaryMatrix) -> list[Circuit]:

        all_circs = self.fix_all_angles(circuit)

        inst_circs = await get_runtime().map(
            Circuit.instantiate,
            all_circs,
            target=target
        )
        
        costs = [self.cost.calc_cost(inst_circ, target=target) for inst_circ in inst_circs]

        valid_circs = [circ for circ, cost in zip(all_circs, costs) if cost <= self.success_threshold]

        return valid_circs

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
                    else:
                        frac_z_cost += 4 * op.gate.k - 4

            rz_cost_per_gate = np.ceil(-1 * np.log10(self.success_threshold)) * 10
            return num_t + num_rz * rz_cost_per_gate + frac_z_cost

        best_circuit = min(circuits, key=cost_fn)
        return best_circuit       
        
    async def run(self, circuit: Circuit, data: PassData) -> None:
        new_circs = await self.exhaustive_search(circuit, target=data.target)
        if len(new_circs) == 0:
            return
        best_circ = self.choose_best_circuit(new_circs, data.target)
        circuit.become(best_circ)