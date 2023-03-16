# graph2tac
graph2tac converts graphs to tactics

# prerequisites on Linux

- standard linux c++ compiler build chain (tested on g++)

- capnproto https://capnproto.org : need capnp proto compiler, the library and the headers. On Ubuntu (Debian based distros) you can install
```
capnproto libcapnp-dev
```
On RPM based distros install
```
capnproto libcapnp libcapnp-devel
```

- CUDA/GPU (optional, if you want to use GPU): please follow https://www.tensorflow.org/install to install on your system.

# installation 

The python `graph2tac` package can be installed with `pip install` in a standard way from git repo (we aren't yet on pypi.org). For developers a recommended way is `pip install -e .` from the cloned source repository which allows to develop directly on the source of the installed package. We recommend and have tested on Python 3.10.

# entry-points

- See `g2t-train-tfgnn --help` to train the TF-GNN model
- See `g2t-server --help` to launch the python server for evaluation
- See `g2t-tfgnn-predict-graphs --help` to evaluate prediction tfgnn networks on supervised data, and plot results

Get started training a simple, small model with the following command:
```
g2t-train-tfgnn --data-dir tests/data/mini_stdlib/dataset/  --dataset-config graph2tac/tfgnn/default_dataset_config.yml --prediction-task-config graph2tac/tfgnn/default_global_argument_prediction.yml --trainer-config graph2tac/tfgnn/default_trainer_config.yml --run-config graph2tac/tfgnn/default_run_config.yml --definition-task-config graph2tac/tfgnn/default_definition_task.yml --log model/
```
You can then start a prediction server that communicates with Coq as follows:
```
g2t-server --arch tfgnn --tcp --port 33333 --host 0.0.0.0 --model model/ --log_level=info --tf_log_level=critical --tactic_expand_bound=8 --total_expand_bound=10 --search_expand_bound=4 --update_all_definitions
```
See https://github.com/coq-tactician/coq-tactician-reinforce for instructions on how to call the server from Coq.
