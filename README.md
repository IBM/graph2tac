# graph2tac
graph2tac converts graphs to tactics

# prerequisites on Linux

- standard linux c++ compiler build chain (tested on g++)

- capnproto https://capnproto.org : need capnp proto compiler, the library and the headers. On Ubuntu you can install `capnproto` and `libcapnp-dev`.

- CUDA/GPU (optional, if you want to use GPU): please follow https://www.tensorflow.org/install to install on your system.

# installation 

The python `graph2tac` package can be installed with `pip install` in a standard way from git repo (we aren't yet on pypi.org). For developers a recommended way is `pip install -e .` from the cloned source repository which allows to develop directly on the source of the installed package. Our key (recommended/tested) pip dependencies are `python>=3.9`,  `tensorflow>=2.8` and `tensorflow-gnn>=0.2.0.dev1`.

# entry-points

- See `g2t-train --help` to train the model
- See `g2t-train-tfgnn --help` to train the TF-GNN model
- See `g2t-server --help` to launch the python server for evaluation

