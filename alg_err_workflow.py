from sys import argv
from bqskit.compiler.compile import compile
from bqskit.compiler.compiler import Compiler
from bqskit.ft.cliffordrz.cliffordrzmodel import CliffordRZModel
from bqskit.ft.cliffordt.cliffordtmodel import CliffordTModel
from bqskit.ft.ftpasses.dpf import DPFPass
from bqskit.ft.ftpasses.convert_frz import ConvertFractionalRZPass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.passes.partitioning import ScanPartitioner, QuickPartitioner
from bqskit.passes.util.unfold import UnfoldPass

def generate_partition_workflow(circ: Circuit, block_size: int = 4) -> list:
    if circ.num_qudits > 30 or circ.num_operations > 600:
        partitioner = QuickPartitioner(block_size)
    else:
        partitioner = ScanPartitioner(block_size)

    workflow = [
        partitioner
    ]

    return workflow

def partition_circ(circ: Circuit, compiler: Compiler, block_size: int = 4) -> Circuit:
    ''' Partitions the circuit into blocks and returns the number of blocks'''
    workflow = generate_partition_workflow(circ, block_size)
    part_circ = compiler.compile(circ, workflow)
    return part_circ

def full_rz_compile(circ: Circuit, compiler: Compiler, 
                    error: float = 1e-3) -> Circuit:
    # Compile to Clifford RZ
    eps_per_gate = error / (circ.num_params + 1)
    rz_circ = compile(circ, model=CliffordRZModel(circ.num_qudits,
                                                epsilon_per_gate=eps_per_gate), 
                            compiler=compiler)
    
    # Use greedy NTRO
    part_circ = partition_circ(rz_circ, compiler)
    workflow = [
        ForEachBlockPass(
            [
                DPFPass(success_threshold=error / part_circ.num_operations),
                ConvertFractionalRZPass()
            ]
        ),
        UnfoldPass()
    ]
    final_rz_circ = compiler.compile(part_circ, workflow)

    return final_rz_circ

def compile_rz_to_clifft(rz_circ: Circuit, compiler: Compiler, 
                         success_threshold: float) -> Circuit:
    num_rz_gates = rz_circ.count(RZGate())
    err_per_gate = success_threshold / num_rz_gates if num_rz_gates > 0 else success_threshold
    cliff_t_model = CliffordTModel(rz_circ.num_qudits, err_per_gate=err_per_gate)
    return compile(rz_circ, model=cliff_t_model, compiler=compiler)

def small_circuit() -> Circuit:
    from bqskit.ir.gates import HGate, CNOTGate, TGate
    circ = Circuit(5)
    circ.append_gate(HGate(), [0])
    circ.append_gate(TGate(), [0])
    circ.append_gate(CNOTGate(), [0, 1])
    circ.append_gate(CNOTGate(), [1, 2])
    circ.append_gate(HGate(), [1])
    circ.append_gate(CNOTGate(), [2, 3])
    circ.append_gate(HGate(), [3])
    circ.append_gate(HGate(), [2])
    circ.append_gate(CNOTGate(), [3, 4])
    circ.append_gate(TGate(), [1])
    circ.append_gate(HGate(), [1])
    circ.append_gate(HGate(), [0])
    return circ

if __name__ == '__main__':
    input_file = argv[1]
    prec = int(argv[2]) # 3 = 10-3 error across circuit
    # Set to -1 for full width pauli products, otherwise pass in a max width
    # and Clifford + T gates
    # max_width = int(argv[3]) if len(argv) > 3 else -1 

    max_width = 2
    error = 10 ** (-prec)

    # Load the circ
    circ = Circuit.from_file(input_file)
    # circ = small_circuit()
    compiler = Compiler(num_workers=-1)

    # Compile to Clifford RZ
    print("Original circuit gate counts:")
    print(circ.gate_counts)
    rz_circ = full_rz_compile(circ, compiler, error=error)
    print("RZ circuit gate counts:")
    print(rz_circ.gate_counts)

    # Compile to Clifford T
    cliff_t_circ = compile_rz_to_clifft(rz_circ, compiler, success_threshold=error)

    print("Clifford+T circuit gate counts:")
    print(cliff_t_circ.gate_counts)

    # Save the final circuit
    output_file = input_file.replace(".qasm", f"_clifft_{prec}.qasm")
    cliff_t_circ.save(output_file)

