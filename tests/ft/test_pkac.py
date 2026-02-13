from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import HGate, CNOTGate, XGate
from bqskit.ft.gadgets.qft import QFTGadget
from bqskit.ft.gates.gidney_adder import GidneyAdder
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.qis.state.state import StateVector
from numpy import allclose, pi


def construct_pkac_circuit(n: int) -> Circuit:
    '''
    Constructs the circuit for the PKAC protocol 
    using the Gidney adder and QFT gadget.

    The circuit is structured as follows:
    1. Prepare the input state by applying H gates to the first n qubits.
    2. Apply a QFT to the ancilla register (qubits 2n to 3n - 1).
    3. Use a Gidney adder to add the input registers 
    (qubits n to 2n - 1 and qubits 2n to 3n - 1) using an ancilla register as
    well.

    This test should add apply Rz(pi) to qubit 0, Rz(pi/2) to qubit 1,
    and Rz(pi/4) to qubit 2, which can be verified by looking 
    at the output state.

    Args:
        n (int): The size of the input registers for the adder.

    Returns:
        Circuit: The constructed PKAC circuit.
    '''
    # 0->n-1: state qubits
    # n->2n- 1 - register A input for adder
    # 2n->3n-1  - register B input for adder (6 is LSB)
    # 3n->4n-2 - ancilla for adder
    c = Circuit(4 * n - 1)
    for i in range(n):
        c.append_gate(HGate(), [i]) 
        # Apply CNOT onto bottom bit of the input and the first ancilla
        c.append_gate(CNOTGate(), [i, n + i])

    # Generate QFT state for the bottom n ancilla, LSB is 6
    c.append_gate(XGate(), [2 * n + (n - 1)])
    # c.append_gate(generate_qft(n), list(range(2 * n, 3 * n)))
    c.append_circuit(QFTGadget.generate(n), list(range(2 * n, 3 * n)))
    # Apply an adder to add the inputs
    # c.append_gate(generate_adder(n), list(range(n, 3 * n)))
    c.append_gate(GidneyAdder(n), list(range(n, 4 * n - 2)))

    for i in range(n):
        # Apply CNOT onto bottom bit of the input and the first ancilla
        c.append_gate(CNOTGate(), [i, n + i])
        c.append_gate(HGate(), [i]) 

    return c

class TestCompileDefaults:
    def test_gidney_adder(self) -> None:
        N = 3
        circuit = Circuit(3*N - 1)
        circuit.append_gate(GidneyAdder(N), list(range(3*N - 1)))
        # Loop over all possible inputs
        for i in range(2 ** (2 * N)):
            circuit = Circuit(3*N - 1)
            # Calculate a and b from i
            a = i >> N
            b = i & (2 ** N - 1)
            # Generate input state for a and b
            circuit = Circuit(3 * N - 1)
            for j in range(N):
                # O is MSB, N - 1 is LSB
                if (a >> j) & 1:
                    circuit.append_gate(XGate(), [N - 1 - j])
                # n is LSB, 2n - 1 is MSB
                if (b >> j) & 1:
                    circuit.append_gate(XGate(), [2 * N - 1 - j])

            circuit.append_circuit(GidneyAdder(N), list(range(3 * N - 1)))

            input = StateVector.zero(3 * N - 1)
            output = circuit.get_statevector(input)

            # Calculate expected output
            expected_output = StateVector.zero(3 * N - 1).numpy()
            expected_index = (a << N) | ((a + b) % (2 ** N))
            # Last n - 1 bits are ancilla
            expected_output[0] = 0
            expected_output[expected_index << (N - 1)] = 1

            assert allclose(output, expected_output, atol=1e-8)

    def test_pkac(self) -> None:
        N = 3
        circuit = construct_pkac_circuit(N)
        in_state = StateVector.zero(4 * N - 1)
        full_out = circuit.get_statevector(in_state)

        all_probs = full_out.get_probs()

        # Get probabilities for first N qubits
        final_probs = []
        for qubit_ind in range(N):
            # Use MSB to get probability of qubit being 1
            prob = sum(prob for i, prob in enumerate(all_probs) if 
                       (i & (1 << (4*N - 2 - qubit_ind))) != 0)
            final_probs.append(prob)

        # Now calculate expected probabilities
        expected_probs = [0] * N
        # We are doing a HZH circuit on qubit 0
        single_in = StateVector.zero(1)
        hzh_circ = Circuit(1)
        hzh_circ.append_gate(HGate(), [0])
        hzh_circ.append_gate(RZGate(), [0], [pi])
        hzh_circ.append_gate(HGate(), [0])
        probs = hzh_circ.get_statevector(single_in).get_probs()
        expected_probs[0] = probs[1]  # Probability of qubit being 1

        # We are doing H Rz(pi/2) H on qubit 1
        single_in = StateVector.zero(1) 
        hzh_circ = Circuit(1)
        hzh_circ.append_gate(HGate(), [0])
        hzh_circ.append_gate(RZGate(), [0], [pi / 2])
        hzh_circ.append_gate(HGate(), [0])
        probs = hzh_circ.get_statevector(single_in).get_probs()
        expected_probs[1] = probs[1]  # Probability of qubit being 1

        # We are doing H Rz(pi/4) H on qubit 2        
        single_in = StateVector.zero(1) 
        hzh_circ = Circuit(1)
        hzh_circ.append_gate(HGate(), [0])
        hzh_circ.append_gate(RZGate(), [0], [pi / 4])
        hzh_circ.append_gate(HGate(), [0])
        probs = hzh_circ.get_statevector(single_in).get_probs()
        expected_probs[2] = probs[1]  # Probability of qubit being 1

        assert allclose(final_probs, expected_probs, atol=1e-6)