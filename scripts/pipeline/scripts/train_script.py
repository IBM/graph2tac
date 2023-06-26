"""
Script governing all training runs

Compute intensive code to be run on a compute node
"""
import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import yaml


@dataclass
class BaseTrainer:
    round: int
    conda_env_path: Path
    paramsdir: Path
    datadir: Path
    resultsdir: Path
    workdir: Path
    model_params: dict[str, Any]

    def save_params_and_build_train_cmd(self) -> list[Any]:
        raise NotImplemented

    def train_in_conda(self) -> int:
        # print gpu information for debugging
        os.system("nvidia-smi")
        
        train_cmd = self.save_params_and_build_train_cmd()
        
        # to run something in conda, put it in a script to set the env variables
        script = []
        script.append("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib")
        script.append("")
        script.append(" ".join(str(c) for c in train_cmd))
        script.append("")
        
        script = "\n".join(script)
        print(script)

        script_sh = self.workdir / "run_training.sh"
        script_sh.write_text(script)

        conda_run_prefix = f"conda run -p {self.conda_env_path} --live-stream"
        exit_code = os.system(f"{conda_run_prefix} sh {script_sh}")
        return exit_code

    def cleanup(self):
        raise NotImplemented


class HModelTrainer(BaseTrainer):
    def save_params_and_build_train_cmd(self) -> list[Any]:
        cmd = []
        cmd.extend(["g2t-train-hmodel", self.datadir])
        cmd.extend(["--max_subgraph_size", self.model_params["max_subgraph_size"]])
        if self.model_params["with_context"]:
            cmd.extend(["--with_context"])
        return cmd
    
    def cleanup(self):
        # move the results
        # There would be at most one hmodel file in the workdir.
        # Move it to its own directory undert the results dir.
        hmodel_file = self.workdir / "hmodel.sav"
        if hmodel_file.exists():
            model_dir = self.resultsdir / "hmodel"
            model_dir.mkdir(exist_ok=True)
            new_hmodel_file = model_dir / "hmodel.sav"
            hmodel_file.rename(new_hmodel_file)
            print(f"hmodel model moved to {model_dir}")


class TF2Trainer(BaseTrainer):

    def save_params_and_build_train_cmd(self) -> list[Any]:
        cmd = []

        # yaml params
        tf2_param_file = self.paramsdir / "tf2_params.yml"
        with tf2_param_file.open(mode="w") as f:
            yaml.dump(self.model_params, f)
        cmd.extend(["g2t-train", self.datadir, tf2_param_file])

        # other arguments
        cmd.extend(["--work-dir", self.workdir])
        cmd.extend(["--output-dir", self.resultsdir])
        cmd.extend(["--logging-level", "INFO"])

        # See if there is already a checkpoint.  If so, then restart from that.
        weights_dir = self.resultsdir / "weights"
        if weights_dir.exists():
            checkpoints_epochs = [
                int(cp.name.split("epoch")[1])
                for cp in weights_dir.iterdir()
                if cp.is_dir() and cp.name.startswith("checkpoint__epoch")
            ]
            if checkpoints_epochs:
                epoch = max(checkpoints_epochs)
                print(f"Found checkpoint epoch {epoch}.  Continuing from that checkpoint.")
                print()
                cmd.extend(["--from-checkpoint", epoch])
        
        return cmd

    def cleanup(self):
        pass 


class TFGNNTrainer(BaseTrainer):

    def save_params_and_build_train_cmd(self) -> list[Any]:
        cmd = []
        # train the tfgnn model
        cmd.extend(["g2t-train-tfgnn"])
        cmd.extend(["--data-dir", self.datadir])

        for config_key in ["dataset", "prediction_task", "definition_task", "trainer", "run"]:
            # save config as a yaml file to the params directory
            config_yml = self.paramsdir / (config_key + ".yml")
            with config_yml.open(mode="w") as f:
                yaml.dump(self.model_params[config_key], f)
            # set cmd arg
            cmd.extend([f"--{config_key.replace('_', '-')}-config", config_yml])

        cmd.extend(["--log", self.resultsdir])
        cmd.extend(["--gpu", "all"])
        
        return cmd

    def cleanup(self):
        pass 


def copy_data_to_tmp_dir(datadir: Path, tmpdir: Path) -> Path:
    assert datadir.is_dir(), datadir
    assert tmpdir.is_dir(), tmpdir
    new_datadir = tmpdir / datadir.name
    shutil.copytree(datadir, tmpdir / datadir.name)
    return new_datadir

def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-arch",
        type=str,
        help="Model architecture: tfgnn, tf2, hmodel"
    )
    parser.add_argument(
        "--model-params",
        type=Path,
        help="YAML parameter file for the model"
    )
    parser.add_argument(
        "--conda_env_path",
        type=Path,
        help="Location of conda environment with model training code loaded"
    )
    parser.add_argument(
        "--round",
        type=int,
        help="Number of training restarts (starting at zero)"
    )
    parser.add_argument(
        "--datadir",
        type=Path,
        help="Location of dataset"
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        help="Directory to run from (files won't be saved)"
    )
    parser.add_argument(
        "--resultsdir",
        type=Path,
        help="Location to store trained models and other results"
    )
    parser.add_argument(
        "--paramsdir",
        type=Path,
        help="Location to store parameter yaml files and other communication files"
    )
    args = parser.parse_args()

    # check inputs
    assert args.model_params.is_file(), f"{args.params} must be an existing file"
    assert str(args.model_params).endswith(".yaml"), f"{args.params} must be a .yaml file"

    return args

def main():
    args = read_args()
    
    with Path(args.model_params).open() as f:
        params = yaml.safe_load(f)

    with tempfile.TemporaryDirectory() as tmpdirname:
        datadir = copy_data_to_tmp_dir(datadir=args.datadir, tmpdir=Path(tmpdirname))
        
        if args.model_arch == "hmodel":
            Trainer = HModelTrainer
        elif args.model_arch == "tf2":
            Trainer = TF2Trainer
        elif args.model_arch == "tfgnn":
            Trainer = TFGNNTrainer
        else:
            raise Exception(f"Unknown model arch: {args.model_arch}")
        
        trainer = Trainer(
            round=args.round,
            conda_env_path=args.conda_env_path,
            paramsdir=args.paramsdir,
            datadir=datadir,
            resultsdir=args.resultsdir,
            workdir=args.workdir,
            model_params=params
        )
        exit_code = trainer.train_in_conda()
        trainer.cleanup()

    if exit_code:
        print()
        print(f"Training did not exit properly.  Exit code: {exit_code}")
    exit(exit_code)


if __name__ == "__main__":
    main()
