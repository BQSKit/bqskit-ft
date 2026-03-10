"""This file tests the GidneyAdder gate and PKAC Rz gate construction."""
from __future__ import annotations

from numpy import allclose
from numpy import pi
from numpy import random

from bqskit.compiler.compiler import Compiler
from bqskit.ft.ftpasses.convert_to_pkac import ConvertToPKAC
from bqskit.ft.ftpasses.pkac_to_gates import PKACtoGatesPass
from bqskit.ft.gadgets.qft import QFTGadget
from bqskit.ft.gates.fractional_rz import FractionalRZGate
from bqskit.ft.gates.gidney_adder import GidneyAdder
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import XGate
from bqskit.ir.gates.constant.t import TGate
from bqskit.ir.gates.constant.tdg import TdgGate
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.qis.state.state import StateVector


class TestCompileDefaults:

    def construct_pkac_circuit(n: int) -> Circuit:
        '''
        Constructs the circuit for the PKAC protocol
        using the Gidney adder and QFT gadget.

        The circuit is structured as follows:
        1. Prepare the input state by applying H gates to the first n qubits.
        2. Apply a QFT to the ancilla register (qubits 2n to 3n - 1).
        3. Use a Gidney adder to add the input registers
        (qubits n to 2n - 1 and qubits 2n to 3n - 1) using an ancilla register
        as well.

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

        state_qubits = list(range(n))
        input_a = list(range(n, 2 * n))
        input_b = list(range(2 * n, 3 * n))
        ancilla = list(range(3 * n, 4 * n - 1))

        for i, q in enumerate(state_qubits):
            c.append_gate(HGate(), [q])
            # Apply CNOT onto a different wire of input A. This applies
            # RZ(pi) on 0th qubit, RZ(pi/2) on 1st qubit, RZ(pi/4) on 2nd qubit,
            # and so forth
            c.append_gate(CNOTGate(), [q, input_a[i]])

        # Generate QFT state for input b
        c.append_gate(XGate(), [input_b[-1]])
        # c.append_gate(generate_qft(n), list(range(2 * n, 3 * n)))
        c.append_circuit(QFTGadget.generate(n), input_b)
        # Apply an adder to add the inputs
        # c.append_gate(generate_adder(n), list(range(n, 3 * n)))
        c.append_gate(
            GidneyAdder(n, add_reset=False),
            input_a + input_b + ancilla,
        )

        for i in range(n):
            # Apply CNOT onto bottom bit of the input and the first ancilla
            c.append_gate(CNOTGate(), [i, n + i])
            c.append_gate(HGate(), [i])

        return c

    def test_gidney_adder(self) -> None:
        N = 3
        # Loop over all possible inputs
        for i in range(2 ** (2 * N)):
            circuit = Circuit(3 * N - 1)
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

            circuit.append_gate(
                GidneyAdder(N, add_reset=False),
                list(range(3 * N - 1)),
            )

            input = StateVector.zero(3 * N - 1)
            output = circuit.get_statevector(input)

            # Calculate expected output
            expected_output = StateVector.zero(3 * N - 1).numpy
            expected_index = (a << N) | ((a + b) % (2 ** N))
            # Last n - 1 bits are ancilla
            expected_output[0] = 0
            expected_output[expected_index << (N - 1)] = 1

            assert allclose(output, expected_output, atol=1e-8)

    def test_pkac_rotations(self) -> None:
        N = 3
        circuit = TestCompileDefaults.construct_pkac_circuit(N)
        total_qubits = 4 * N - 1
        assert circuit.num_qudits == total_qubits
        in_state = StateVector.zero(total_qubits)
        full_out = circuit.get_statevector(in_state)

        all_probs = full_out.get_probs()

        # Get probabilities for first N qubits
        final_probs = []
        for qubit_ind in range(N):
            # Use MSB to get probability of qubit being 1
            prob = sum(
                prob for i, prob in enumerate(all_probs) if
                (i & (1 << (total_qubits - 1 - qubit_ind))) != 0
            )
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

    def test_fractional_rz_to_t_decomp(self) -> None:
        # Test that the decomposition of FractionalRZGate with k = 3 is correct
        N = 1
        K = 3

        for num in range(4 ** K):
            gate = FractionalRZGate(num, K)
            ang = 2 * num * pi / (2 ** K)
            rz_gate = RZGate()
            target = rz_gate.get_unitary([ang])
            un_dist = target.get_distance_from(gate.get_unitary())
            assert allclose(un_dist, 0, atol=1e-7)
            frac_circ = gate.get_circuit(num, K)
            circ_dist = target.get_distance_from(frac_circ.get_unitary())
            assert allclose(circ_dist, 0, atol=1e-7)

    def test_pkac_decomposition(self) -> None:
        # Contstruct a circuit with random FractionalRZGates and then
        # apply the PKAC decomposition to it. Then verify that the output
        # state is the same as the original circuit.
        N = 3
        K = 4
        num_layers = 3
        rzs_per_layer = 2

        circ = Circuit(N)

        for i in range(N):
            circ.append_gate(HGate(), [i])

        for _ in range(num_layers):
            # Choose 3 random qubits and apply random Rzs to them
            rand_qubits = random.choice(N, size=rzs_per_layer, replace=False)
            for i, q in enumerate(rand_qubits):
                num = random.randint(1, 2 ** K)
                circ.append_gate(FractionalRZGate(num, K), [q])

            # Should do some kickbacks
            for i in range(N - 1):
                circ.append_gate(CNOTGate(), [i, i + 1])

        for i in range(N):
            circ.append_gate(HGate(), [i])

        orig_num_rzs = num_layers * rzs_per_layer
        orig_h_gates = circ.count(HGate())

        # Calculate probabilities for original circuit
        in_state = StateVector.zero(N)
        original_out = circ.get_statevector(in_state)
        original_probs = original_out.get_probs()
        expected_probs = []
        for qubit_ind in range(N):
            prob = sum(
                prob for i, prob in enumerate(original_probs) if
                (i & (1 << (N - 1 - qubit_ind))) != 0
            )
            expected_probs.append(prob)

        # Now, convert to PKAC
        workflow = [
            ConvertToPKAC(ks = [K], add_measurements=False, 
                          use_catalyzed_sqrt_t=False),
        ]

        with Compiler() as compiler:
            pkac_circ = compiler.compile(circ, workflow=workflow)
  
        # Calculate probabilities for PKAC circuit
        in_state = StateVector.zero(pkac_circ.num_qudits)
        pkac_out = pkac_circ.get_statevector(in_state)
        pkac_probs = pkac_out.get_probs()
        final_probs = []
        for qubit_ind in range(N):
            prob = sum(
                prob for i, prob in enumerate(pkac_probs) if
                (i & (1 << (pkac_circ.num_qudits - 1 - qubit_ind))) != 0
            )
            final_probs.append(prob)

        assert allclose(final_probs, expected_probs, atol=1e-6)

        # Now test decomposition of PKAC circuit
        workflow = [
            PKACtoGatesPass(),
        ]

        with Compiler() as compiler:
            final_circ = compiler.compile(pkac_circ, workflow=workflow)

        # Now let's see if the number of Ts and H gates are correct.
        num_ts_from_gidney = 4 * K - 4
        num_ts_from_qft = 3 * K - 3  # Is this correct? TODO: Mathias
        expected_num_ts = orig_num_rzs * (num_ts_from_gidney) + num_ts_from_qft

        num_ts = final_circ.count(TGate()) + final_circ.count(TdgGate())

        assert num_ts == expected_num_ts

        num_hs_from_gidney = (K - 1) * 3  # Each ancilla gets 3 H gates
        num_hs_from_qft = K  # QFT gadget has 1 H gate per qubit # TODO: Mathias

        expected_num_hs = orig_num_rzs * \
            (num_hs_from_gidney) + num_hs_from_qft + orig_h_gates

        num_hs = final_circ.count(HGate())

        assert num_hs == expected_num_hs

    def test_pkac_decomposition_cat_sqrt_t(self) -> None:
        # Contstruct a circuit with random FractionalRZGates and then
        # apply the PKAC decomposition to it. Then verify that the output
        # state is the same as the original circuit.
        N = 4
        K = 4
        num_layers = 3
        rzs_per_layer = 2

        circ = Circuit(N)

        for i in range(N):
            circ.append_gate(HGate(), [i])

        for _ in range(num_layers):
            # Choose 3 random qubits and apply random Rzs to them
            rand_qubits = random.choice(N, size=rzs_per_layer, replace=False)
            for i, q in enumerate(rand_qubits):
                # num = random.randint(1, 2 ** K)
                num = 3
                # Reduce num 
                actual_k = K
                while num % 2 == 0 and actual_k > 0:
                    num //= 2
                    actual_k -= 1
                circ.append_gate(FractionalRZGate(num, actual_k), [q])

            # Should do some kickbacks
            for i in range(N - 1):
                circ.append_gate(CNOTGate(), [i, i + 1])

        for i in range(N):
            circ.append_gate(HGate(), [i])

        # Calculate probabilities for original circuit
        in_state = StateVector.zero(N)
        original_out = circ.get_statevector(in_state)
        original_probs = original_out.get_probs()
        expected_probs = []
        for qubit_ind in range(N):
            prob = sum(
                prob for i, prob in enumerate(original_probs) if
                (i & (1 << (N - 1 - qubit_ind))) != 0
            )
            expected_probs.append(prob)

        # Now, convert to PKAC
        workflow = [
            ConvertToPKAC(ks = [3, 4], add_measurements=False, 
                        use_catalyzed_sqrt_t=True),
        ]

        with Compiler() as compiler:
            pkac_circ = compiler.compile(circ, workflow=workflow)

        # Should use catalyzed sqrt_t instead of PKAC
        assert pkac_circ.num_qudits <= (N + 3) # 3 ancilla for sqrt_t

        print("Original circuit:", circ.num_qudits, circ.gate_counts)
        print("PKAC circuit:", pkac_circ.num_qudits, pkac_circ.gate_counts)

        # Calculate probabilities for PKAC circuit
        in_state = StateVector.zero(pkac_circ.num_qudits)
        pkac_out = pkac_circ.get_statevector(in_state)
        pkac_probs = pkac_out.get_probs()
        final_probs = []
        for qubit_ind in range(N):
            prob = sum(
                prob for i, prob in enumerate(pkac_probs) if
                (i & (1 << (pkac_circ.num_qudits - 1 - qubit_ind))) != 0
            )
            final_probs.append(prob)

        print(final_probs)
        print(expected_probs)

        assert allclose(final_probs, expected_probs, atol=1e-6)