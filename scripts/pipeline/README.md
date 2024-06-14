# Pipeline for training and testing

These are a series of scripts for running training and benchmarking.

## Why are they so complicated?

They are designed for a cluster system which kills jobs after 24 hours, so it makes it difficult to process the results of a job or start a new job if the previous one didn't finish.  Right now the code manually waits and checks until a job finishes (but maybe it will be modified with support for automatically starting another job after one finishes using built in support for this in the cluster system).

## How do I use this system?
(These steps are subject to improvement.)

### Prerequistes
- A fairly new version of python with yaml installed.
- Conda/Miniconda
- Maybe a few more things I don't realize

### Steps to run
- Create a directory of the form `YYYY-MM-DD_...`, e.g. `2023-02-23_A_train_model`, and paste in the scripts directory into it, e.g. `2023-02-23_A_train_model/scripts`.  (TODO: Make it runnable from another directory.)
- Put in a parameter file labeled `params.yaml` into your directory.  (More on this below.)
- From within `scripts` run `sh run.sh` for short jobs or jobs which are run inside a job system or `sh tmux_run.sh` for long jobs where the outer job executer is running in `tmux`.

### What is happening when I run this?
Depending on your `params.yaml` it runs training or benchmarking.  (Support for automatic benchmarking is still to be added.)  The scripts add the following subdirectories:
- logs: This is where all the logs for the various jobs are stored.
- results: This stores the results of the run including trained model weights and tensorboard files
- params: This is for communication between jobs.
- workdir: This is all the stuff which can be deleted after all the jobs end like conda environments, downloaded repos, etc.

For training, it starts by installing a fresh conda environment and running training from that inside a job/tmux depending on your settings.  After the training finishes or the job is killed, it uploads the results and continues training if needed.  Further plans also include better summarization of the benchmark results.

For benchmarking, it starts by installing a fresh conda environment and opam switch and running benchmarking from that environment/switch inside a job/tmx depending on your settings.  Currently the benchmark jobs run sequentially.  Further plans include running all the benchmarks in parallel, and supporting long benchmarks which take over 24 hours (where the job will get killed in the cluster this is designed for).  Further plans also include better summarization of the benchmark results.

## Parameter files

TODO: Include examples of a training parameter file and a benchmarking parameter file.

## Adding support for another cluster like SLURM
If you have no time limit on jobs, currently the scripts run sequentially and it is easy to just run from within a job using the `tmux` settings in the parameter files (as in the examples).  But it is also possible to add in your own job system fairly easily to the scripts.