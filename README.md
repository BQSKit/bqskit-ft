# BQSKit-FT: Fault-Tolerant Quantum Compilation

A BQSKit extension package for compiling quantum circuits to fault-tolerant gate sets.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Overview

BQSKit-FT extends the Berkeley Quantum Synthesis Toolkit ([BQSKit](https://github.com/BQSKit/bqskit)) with specialized compilation workflows and machine models for fault-tolerant quantum computing. This package provides tools for compiling arbitrary quantum circuits into fault-tolerant gate sets such as Clifford+T and Clifford+RZ.

## Key Features

### Machine Models
- **CliffordTModel**: Fault-tolerant machine model with Clifford+T gate set
- **CliffordRZModel**: Fault-tolerant machine model with Clifford+RZ gate set
- **FaultTolerantModel**: Base class for custom fault-tolerant machine models

### Synthesis Passes
- **GridSynthPass**: High-precision RZ gate synthesis using the gridsynth algorithm
- **RoundToDiscreteZPass**: Rounds RZ gates to discrete Clifford+T equivalents
- **IsolateRZGatePass**: Isolates RZ gates for individual processing

### Compilation Workflows
- Pre-built workflows for circuit, unitary, state preparation, and state mapping compilation
- Support for multiple optimization levels (1-4)
- Configurable synthesis precision and error thresholds
- Optional RZ gate decomposition into Clifford+T

## Installation

### Dependencies
```bash
pip install bqskit numpy scipy pygridsynth
```

### Install BQSKit-FT
```bash
git clone https://github.com/BQSKit/bqskit-ft.git
cd bqskit-ft
pip install -e .
```

## Quick Start

### Basic Clifford+T Compilation

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
model = CliffordTModel(2)

# Compile to Clifford+T gate set
ft_circuit = compile(circuit, model)

# Verify output uses only fault-tolerant gates
print(f"Gate set: {ft_circuit.gate_set}")
```

### High-Precision RZ Synthesis

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

### Custom Workflows

```python
from bqskit.ft.ftpasses import IsolateRZGatePass, GridSynthPass
from bqskit.passes import ForEachBlockPass, UnfoldPass

# Build custom workflow for mixed circuits
workflow = [
    IsolateRZGatePass(),                    # Isolate RZ gates
    ForEachBlockPass([GridSynthPass(15)]),  # Synthesize each RZ gate
    UnfoldPass(),                           # Flatten the circuit
]

with Compiler() as compiler:
    result = compiler.compile(circuit, workflow)
```

## Machine Models

### CliffordTModel
Represents a fault-tolerant quantum computer with the Clifford+T gate set:
- **Clifford gates**: H, X, Y, Z, S, S†, √X, CNOT, CZ, SWAP
- **Non-Clifford gates**: T, T†, RZ

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
- **Clifford gates**: H, X, Y, Z, S, S†, √X, CNOT, CZ, SWAP
- **Non-Clifford gates**: T, T†, RZ (RZ gates preserved)

```python
from bqskit.ft import CliffordRZModel

model = CliffordRZModel(num_qudits=3)
```

## Synthesis Passes

### GridSynthPass
Implements the gridsynth algorithm for optimal Clifford+T synthesis of RZ gates:

```python
from bqskit.ft.ftpasses import GridSynthPass

# Precision: 10^-15 approximation error
gridsynth = GridSynthPass(precision=15)
```

**Features:**
- Arbitrary precision synthesis using mpmath
- Provably optimal T-count for single-qubit unitaries
- Configurable precision (affects T-gate count vs. accuracy trade-off)

### RoundToDiscreteZPass
Rounds RZ gates to the nearest π/4 multiple (Clifford+T equivalent):

```python
from bqskit.ft.ftpasses import RoundToDiscreteZPass

rounder = RoundToDiscreteZPass(synthesis_epsilon=1e-8)
```

**Use cases:**
- Fast approximation for near-Clifford+T angles
- Pre-processing step before gridsynth
- Error-tolerant applications

## Compilation Options

### Optimization Levels
- **Level 1**: Fast compilation, basic optimization
- **Level 2**: Balanced speed/quality
- **Level 3**: Aggressive optimization
- **Level 4**: Maximum optimization (slowest)

### Synthesis Parameters
- `synthesis_epsilon`: Maximum unitary distance error (default: 1e-8)
- `max_synthesis_size`: Maximum block size for synthesis (default: 3)
- `decompose_rz`: Whether to decompose RZ gates to Clifford+T (default: True)

```python
from bqskit import compile
from bqskit.ft import CliffordTModel

model = CliffordTModel(2)

# High-precision, aggressive optimization
result = compile(
    circuit,
    model,
    optimization_level=4,
    synthesis_epsilon=1e-12,
    max_synthesis_size=4
)
```

## Advanced Usage

### Custom Gate Sets
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
