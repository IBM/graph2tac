"""
Script governing all training runs

Compute intensive code to be run on a compute node
"""
import datetime
import os
from pathlib import Path
import shutil
import urllib
import yaml

import build_model_spec
from job_runner import JobRunner
from utils import Utils


class TrainPipeline:
    run_dir_name: str
    params: dict
    dirs: dict[str, Path]
    conda_env_path: Path

    def __init__(self, run_name: str, params: dict, outer_directories: dict[str, Path]):
        self.run_dir_name = run_name
        self.params = params
        self.dirs = outer_directories.copy()
    
    def setup_train_dirs(self):
        print()
        print("========")
        print("Setting up directories for training")
        print("========")
        print()

        # specialize some directories for training
        for dir in ["params", "results", "workdir"]:
            self.dirs[dir] = self.dirs[dir] / "train"
            print(f"Make {self.dirs[dir]}")
            self.dirs[dir].mkdir(exist_ok=True)

        print(f"Setting {self.dirs['workdir']} as workdir")
        os.chdir(self.dirs["workdir"])
    
    def setup_conda_env(self):
        self.conda_env_path = self.dirs["workdir"] / "venv"
        Utils.setup_conda(self.conda_env_path, self.params["conda_env"])

    def extract_model_checkpoints(self) -> list[tuple[int, Path]]:
        """Retrieve all available model checkpoint directories,
        possibly moving files to results directory if needed.

        :return: List of all available checkpoints as (epoch, directory path) pairs
        """
        model_type = self.params["model"]

        if model_type == "hmodel":
            # There would be at most one hmodel results.
            # It will have already been moved to this directory.
            model_dir = self.dirs["results"] / "hmodel"
            if model_dir.exists():
                return [(0, model_dir)]
            else:
                return []
        
        elif model_type == "tf2":
            weights_dir = self.dirs["results"] / "weights"
            checkpoints = [
                (int(cp.name.split("epoch")[1]), cp)
                for cp in weights_dir.iterdir()
                if cp.is_dir() and cp.name.startswith("checkpoint__epoch")
            ]
            return checkpoints
        
        elif model_type == "tfgnn":
            # the results directory is the whole checkpoint,
            # so for now just return the max checkpoint
            weights_dir = self.dirs["results"] / "ckpt"
            checkpoints = [
                int(cp.name.split(".")[0].split("ckpt-")[1])
                for cp in weights_dir.iterdir()
                if cp.name.startswith("ckpt-")
            ]
            if checkpoints:
                return [(max(checkpoints), self.dirs["results"])]
        
        raise Exception(f"Not reconized model type: ", model_type)

    def upload_model(self, model_dir, epoch):
        # upload the model to the fileserver via rsync
        fileserver_ssh = Path(self.params["upload_model"]["fileserver_ssh"])
        tar_file_name = Path(f"{self.run_dir_name}_epoch{epoch}.tar.gz")
        os.system(f"tar -czvf {tar_file_name} -C {model_dir.parent} {model_dir.name}")
        md5sum = os.popen(f"md5sum {tar_file_name}").read().split()[0]
        new_tar_file_name = Path(f"{self.run_dir_name}_epoch{epoch}.{md5sum}.tar.gz")
        os.system(f"rsync {tar_file_name} {fileserver_ssh / new_tar_file_name}")

        # use the partial spec to make a full spec
        if self.params["upload_model"]["fileserver_url"]:
            fileserver_url = self.params["upload_model"]["fileserver_url"]
            spec_params = self.params["upload_model"]["spec"].copy()
            spec_params["model_source_url"] = urllib.parse.urljoin(fileserver_url, str(new_tar_file_name))
            spec_params["model_md5sum"] = md5sum
            spec_params["description"] = f"{self.run_dir_name}_epoch{epoch}"

            # TODO check the right architecture is used in the server

            git_branch = self.params["upload_model"]["spec_git_branch"]
            git_commit = build_model_spec.create_and_push_spec(git_branch=git_branch, params=spec_params)

            return git_commit
        else:
            return None

    def train_loop(self):
        # save model params
        model_params = self.params["model_params"]
        model_params_file = self.dirs["params"] / "all_model_params.yaml"
        with model_params_file.open(mode="w") as f:
            yaml.dump(model_params, f)
        
        dataset_dir = Path(self.params["data"]["datadir"]) / self.params["data"]["dataset"]
        prev_epoch = -1
        for round in range(self.params["max_restarts"] + 1):
            print()
            print("========")
            print("Run job to train model")
            print("========")
            print()
            cmd = [
                "python", "-u",
                self.dirs["scripts"] / "train_script.py",
                "--model-arch", self.params["model"],
                "--model-params", model_params_file,
                "--round", round,
                "--conda_env_path", self.conda_env_path,
                "--paramsdir", self.dirs["params"],
                "--datadir", dataset_dir,
                "--resultsdir", self.dirs["results"],
                "--workdir", self.dirs["workdir"],
            ]
            job_runner = JobRunner(
                job_params=self.params["job"],
                log_dir=self.dirs["logs"],
                job_name="train"
            )
            job_runner.run_cmd_and_block(cmd)
            
            checkpoints = self.extract_model_checkpoints()

            if not checkpoints:
                print("Didn't generate any checkpoints.  Stopping run.")
                raise Exception("Didn't generate any checkpoints.")

            print()
            print("========")
            print("Processing result")
            print("========")
            print()
            
            epoch, checkpoint_dir = max(checkpoints)
            if epoch > prev_epoch:
                prev_epoch = epoch
                spec_commit = self.upload_model(checkpoint_dir, epoch=epoch)
            
                # TODO: Create automatic benchmark config
                # TODO: Benchmark checkpoint
            
            else:
                print(f"No new checkpoint created!")
                return  # TODO: There is a better way to check this is done.

            


    def clean_up(self):
        print()
        print("========")
        print("Clean up")
        print("========")
        print()

        print(f"Remove {self.dirs['workdir']}")
        shutil.rmtree(self.dirs['workdir'])

def train(
    run_name: str,
    train_params: dict,
    directories: dict[str, Path]
):
    train_pipeline = TrainPipeline(
        run_name=run_name, 
        params=train_params, 
        outer_directories=directories
    )
    train_pipeline.setup_train_dirs()
    train_pipeline.setup_conda_env()
    train_pipeline.train_loop()
    train_pipeline.clean_up()

def main():
    params = Utils.get_params()
    if "train" not in params:
        print("No train parameters.  No training to do.")
        return

    train(
        run_name=Utils.get_run_dir_name(),
        train_params=params["train"],
        directories=Utils.get_directories()
    )

if __name__ == "__main__":
    main()