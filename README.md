# Graph2Tac
`graph2tac` converts graphs to tactics

## Prerequisites

### Linux
You will need the following dependencies which can be installed with, e.g., `apt-get` or `conda`:
- git, c++ compiler
- python 3.9 and pip
- capnproto

### Intel MacOS

### Other operating systems

Other operating systems are currently not tested/supported.

### Installating `graph2tac`:
Clone this repository, and from within run one of these commands:

`graph2tac` is a python package.  To install:
- Install everything: `pip install .`
- Install tf-gnn version of the code: `pip install . --pre`
- Install in developer mode: `pip install -e .`
- Install tf-gnn version in developer mode: `pip install -e . --pre`

## Primary applications

After installing `graph2tac` the following entry points are available.  Run with `--help` for more information and instructions.
- `g2t-train` trains a model to turn graphs to tactics
- `g2t-server` runs a TCP server to convert graphs to tactics via a communication protocol

## Running tests

To run tests:  `sh tests/check.sh`