# BQSKit-FT: Fault-Tolerant Quantum Compilation

A BQSKit extension package for compiling quantum circuits to fault-tolerant gate sets.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Overview

BQSKit-FT extends the Berkeley Quantum Synthesis Toolkit ([BQSKit](https://github.com/BQSKit/bqskit)) with specialized compilation workflows and machine models for fault-tolerant quantum computing. This package provides tools for compiling arbitrary quantum circuits into fault-tolerant gate sets such as Clifford+T and Clifford+RZ.

## Installation
BQSKit-FT can be installed from PyPI using
```bash
pip install bqskit-ft
```

### Install from github
For the most up to date version install from github using:
```bash
git clone https://github.com/BQSKit/bqskit-ft.git
cd bqskit-ft
pip install -e .
```

## Quick Start

### Basic Clifford+T Compilation
Synthesis to fault-tolerant gate sets is done by specifying a fault-tolerant `MachineModel`. For The Clifford+T gate set, that is the `CliffordTModel`. While the Clifford+RZ gate set is not fault-tolerant, it's useful to have. The Clifford+RZ gate set can be targeted by specifying a `CliffordRZModel`.
```python
from bqskit import Circuit, compile
from bqskit.ft import CliffordTModel
from bqskit.ir.gates import RZGate, CNOTGate

# Create a circuit with arbitrary rotations
circuit = Circuit(2)
circuit.append_gate(CNOTGate(), [0, 1])
circuit.append_gate(RZGate(), [0], [0.12345])  # Arbitrary angle
circuit.append_gate(RZGate(), [1], [0.67890])

# Define fault-tolerant machine model
model = CliffordTModel(2)  # or model = CliffordRZModel(2)

# Compile to Clifford+T gate set
ft_circuit = compile(circuit, model)

# Verify output uses only fault-tolerant gates
print(f"Gate set: {ft_circuit.gate_set}")
```

### High-Precision RZ Synthesis with `pygridsynth`

```python
from bqskit.ft.ftpasses import GridSynthPass
from bqskit.compiler import Compiler

# Single RZ gate with arbitrary angle
circuit = Circuit(1)
circuit.append_gate(RZGate(), [0], [0.1234567890123456])

# High-precision synthesis (20 decimal places)
gridsynth = GridSynthPass(precision=20)

with Compiler() as compiler:
    result = compiler.compile(circuit, [gridsynth])

print(f"Synthesized with {result.num_operations} gates")
```

## Machine Models
### CliffordTModel
Represents a fault-tolerant quantum computer with the Clifford+T gate set:

```python
from bqskit.ft import CliffordTModel

# 4-qubit fault-tolerant machine
model = CliffordTModel(
    num_qudits=4,
    clifford_gates=None,  # Use default Clifford gates
    non_clifford_gates=None,  # Use default T gates + RZ
)
```

### CliffordRZModel
Alternative model that keeps RZ gates (no T gate decomposition):

```python
from bqskit.ft import CliffordRZModel

model = CliffordRZModel(num_qudits=3)
```

## Advanced Usage

### Custom Gate Sets
Other gate sets can be targeted with the following:
```python
from bqskit.ft import FaultTolerantModel
from bqskit.ir.gates import HGate, CNOTGate, TGate

# Custom fault-tolerant model
custom_clifford = [HGate(), CNOTGate()]  # Minimal Clifford set
custom_non_clifford = [TGate()]          # T gates only

model = FaultTolerantModel(
    num_qudits=2,
    clifford_gates=custom_clifford,
    non_clifford_gates=custom_non_clifford
)
```

## References

- [BQSKit Documentation](https://bqskit.readthedocs.io/)
- [Gridsynth Algorithm](https://arxiv.org/abs/1403.2975)

## Citation

If you use `bqskit-ft` in your research, please cite:

```bibtex
@software{bqskit_ft,
  title = {{BQSKit-FT}: Fault-Tolerant Quantum Compilation},
  author = {Weiden, Mathias},
  year = {2024},
  url = {https://github.com/BQSKit/bqskit-ft}
}
```
