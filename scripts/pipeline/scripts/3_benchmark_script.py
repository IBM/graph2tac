import os
from pathlib import Path
import yaml
import shutil
import datetime

from build_model_spec import create_and_push_spec
from job_runner import JobRunner
from utils import Utils


class BenchmarkPipeline:
    """
    Controls the overall benchmark process
    """
    run_dir_name: str
    params: dict
    dirs: dict[str, Path]
    conda_env_path: Path
    opam_switch_path: Path
    benchmark_system: Path

    def __init__(self, run_name: str, params: dict, outer_directories: dict[str, Path]):
        self.run_dir_name = run_name
        self.params = params
        self.dirs = outer_directories.copy()

    def setup_benchmark_dirs(self):
        print()
        print("========")
        print("Setting up directories for benchmarking")
        print("========")
        print()

        # specialize some directories for benchmarking
        for dir in ["params", "results", "workdir"]:
            self.dirs[dir] = self.dirs[dir] / "benchmark"
            print(f"Make {self.dirs[dir]}")
            self.dirs[dir].mkdir(exist_ok=True)

        print(f"Setting {self.dirs['workdir']} as workdir")
        os.chdir(self.dirs["workdir"])

    def setup_conda_opam_env(self):
        # conda
        self.conda_env_path = self.dirs["workdir"] / "venv"
        Utils.setup_conda(self.conda_env_path, self.params["conda_env"])
        conda_run_prefix = f"conda run -p {self.conda_env_path} --live-stream"

        # benchmark_system
        self.benchmark_system = self.dirs["workdir"] / "benchmark-system"
        os.system(f"git clone git@github.com:coq-tactician/benchmark-system.git {self.benchmark_system}")

        # opam
        self.opam_switch_path = self.dirs["workdir"]  # not created yet
        # just hard-code these for now
        opam_env_params = {
            "setup_cmds": ["opam update", f"opam install --yes {self.benchmark_system}"]
        }
        Utils.setup_opam(conda_run_prefix, self.opam_switch_path, opam_env_params)

    def clean_up(self):
        pass


class SingleBenchmarkPipeline:
    """
    Controls a single benchmark
    """
    run_name: str
    benchmark_id: str
    params: dict
    conda_env_path: Path
    opam_switch_path: Path
    benchmark_system: Path

    def __init__(self, run_name: str, ix: int, params: dict, outer_directories: dict, conda_env_path: Path, opam_switch_path: Path, benchmark_system: Path):
        self.run_name=run_name
        self.ix=ix
        self.params=params.copy()
        self.dirs=outer_directories.copy()
        self.conda_env_path=conda_env_path
        self.opam_switch_path=opam_switch_path
        self.benchmark_system=benchmark_system

    def setup_benchmark_dirs(self):
        print()
        print("========")
        print("Setting up directories for benchmarking")
        print("========")
        print()

        # specialize some directories for benchmarking
        for dir in ["params", "results", "workdir"]:
            self.dirs[dir] = self.dirs[dir] / ("benchmark_" + str(self.ix))
            print(f"Make {self.dirs[dir]}")
            self.dirs[dir].mkdir(exist_ok=True)

        print(f"Setting {self.dirs['workdir']} as workdir")
        os.chdir(self.dirs["workdir"])

    def build_and_upload_benchmark_spec(self):
        if "spec_repo" in self.params and "spec_commit" in self.params:
            print("Spec commit given:")
            print(self.params["spec_repo"])
            print()
            return
        
        if "spec_repo" not in self.params:
            self.params["spec_repo"] = "git+ssh://git@github.com/coq-tactician/coq-graph2tac-trained"
        
        #TODO(jrute): Add support for uploading specs to other locations
        assert self.params["spec_repo"] == "git+ssh://git@github.com/coq-tactician/coq-graph2tac-trained", (
            "Don't yet have support for creating specs in other benchmark repos"
        )
        spec_commit = create_and_push_spec(
            git_branch=self.params["spec_branch"],
            params=self.params["spec"]
        )
        self.params["spec_commit"] = spec_commit
    
    def clone_data_repo(self):
        self.datadir = self.dirs["workdir"] / "benchmark-data"
        os.system(f"git clone git@github.com:coq-tactician/benchmark-data.git {self.datadir}")
    
    def update_params(self):
        benchmark_settings = self.params["benchmark_settings"]
        benchmark_settings["benchmark-data"] = str(self.datadir)
        benchmark_settings["benchmark-repo"] = self.params["spec_repo"]
        benchmark_settings["benchmark-commit"] = self.params["spec_commit"]
        benchmark_settings["compile-allocator"] = str(self.benchmark_system / "local" / "compile_allocator")
        benchmark_settings["bench-allocator"] = str(self.benchmark_system / "local" / "bench_allocator")
    
    def remove_data_repo(self):
        shutil.rmtree(self.datadir)
    
    def run_benchmark(self):
        # save model params
        benchmark_params_file = self.dirs["params"] / "benchmark_params.yaml"
        with benchmark_params_file.open(mode="w") as f:
            yaml.dump(self.params, f)
        
        print()
        print("========")
        print("Run job to benchmark model")
        print("========")
        print()

        # TODO(jrute): Handle if need to restart job
        round = 0
        cmd = [
            "python", "-u",
            self.dirs["scripts"] / "benchmark_script.py",
            "--params", benchmark_params_file,
            "--round", round,
            "--conda_env_path", self.conda_env_path,
            "--opam_switch_path", self.opam_switch_path,
            "--datadir", self.datadir,
            "--paramsdir", self.dirs["params"],
            "--resultsdir", self.dirs["results"],
            "--workdir", self.dirs["workdir"],
        ]
        job_runner = JobRunner(
            job_params=self.params["job"],
            log_dir=self.dirs["logs"],
            job_name="benchmark_" + str(self.ix)
        )
        job_runner.run_cmd_and_block(cmd)


def benchmark(
    run_name: str,
    benchmark_params: dict,
    directories: dict[str, Path]
):
    benchmark_pipeline = BenchmarkPipeline(
        run_name=run_name,
        params=benchmark_params,
        outer_directories=directories
    )
    benchmark_pipeline.setup_benchmark_dirs()
    benchmark_pipeline.setup_conda_opam_env()

    for ix, single_benchmark_params in enumerate(benchmark_params["benchmarks"]):
        #TODO(jrute): Make this NOT single threaded
        single_benchmark_pipeline = SingleBenchmarkPipeline(
            run_name=run_name,
            ix=ix,
            params=single_benchmark_params,
            outer_directories=benchmark_pipeline.dirs,
            conda_env_path=benchmark_pipeline.conda_env_path,
            opam_switch_path=benchmark_pipeline.opam_switch_path,
            benchmark_system=benchmark_pipeline.benchmark_system
        )
        single_benchmark_pipeline.setup_benchmark_dirs()
        single_benchmark_pipeline.build_and_upload_benchmark_spec()
        single_benchmark_pipeline.clone_data_repo()
        single_benchmark_pipeline.update_params()
        single_benchmark_pipeline.run_benchmark()
        single_benchmark_pipeline.remove_data_repo()
    
    benchmark_pipeline.clean_up()

def main():
    with Path("../params.yaml").open() as f:
        params = yaml.safe_load(f)
    
    # set up the directories
    if "benchmark" not in params:
        print("No benchmark parameters.  No benchmarking to do.")
        return
    
    benchmark(
        run_name=Utils.get_run_dir_name(),
        benchmark_params=params["benchmark"],
        directories=Utils.get_directories()
    )

if __name__ == "__main__":
    main()