"""
Script governing benchmarking

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
class Benchmarker:
    round: int
    conda_env_path: Path
    opam_switch_path: Path
    paramsdir: Path
    resultsdir: Path
    datadir: Path
    workdir: Path
    params: dict[str, Any]

    def build_script(self) -> str:
        benchmark_settings = self.params["benchmark_settings"].copy()

        # to run something in opam, put it in a script starting with `eval $(opam env)``
        script = []
        script.append("opam update")
        script.append(f"eval $(opam env --switch={self.opam_switch_path})")
        script.append("")
        script.append("export PATH=$CONDA_PREFIX/bin:$PATH")
        script.append("export CPATH=$CONDA_PREFIX/include:$CPATH")
        script.append("")
        script.append("tactician-benchmark \\")
        for setting, value in self.params["benchmark_settings"].items():
            if setting != "coq-project":
                script.append(f"-{setting} {value} \\")
        coq_project = benchmark_settings["coq-project"]
        script.append(f"{coq_project}")
        script.append("")
        
        return "\n".join(script)

    def run_benchmark(self) -> int:
        script = self.build_script()
        
        print(script)

        # to run an opam command, make a temporary script and then run that
        conda_run_prefix = f"conda run -p {self.conda_env_path} --live-stream"
        script_sh = self.workdir / "run_benchmark.sh"
        script_sh.write_text(script)
        exit_code = os.system(f"{conda_run_prefix} sh {script_sh}")

        return exit_code

    def cleanup(self):
        #TODO(jrute): Find way to indicate that benchmark finished successfully
        #probably deleting benchmark results dir
        pass

def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        type=Path,
        help="YAML parameter file for the model"
    )
    parser.add_argument(
        "--conda_env_path",
        type=Path,
        help="Location of conda environment"
    )
    parser.add_argument(
        "--opam_switch_path",
        type=Path,
        help="Location of opam environment"
    )
    parser.add_argument(
        "--round",
        type=int,
        help="Number of restarts (starting at zero)"
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        help="Directory to run from (files won't be saved)"
    )
    parser.add_argument(
        "--datadir",
        type=Path,
        help="Benchmark data"
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
    assert args.params.is_file(), f"{args.params} must be an existing file"
    assert str(args.params).endswith(".yaml"), f"{args.params} must be a .yaml file"

    return args

def main():
    args = read_args()
    
    with Path(args.params).open() as f:
        params = yaml.safe_load(f)

    with tempfile.TemporaryDirectory() as tmpdirname:
        benchmarker = Benchmarker(
            round=args.round,
            conda_env_path=args.conda_env_path,
            opam_switch_path=args.opam_switch_path,
            paramsdir=args.paramsdir,
            resultsdir=args.resultsdir,
            datadir=args.datadir,
            workdir=args.workdir,
            params=params
        )
        exit_code = benchmarker.run_benchmark()
        benchmarker.cleanup()

    if exit_code:
        print()
        print(f"Did not exit properly.  Exit code: {exit_code}")
    exit(exit_code)

if __name__ == "__main__":
    main()