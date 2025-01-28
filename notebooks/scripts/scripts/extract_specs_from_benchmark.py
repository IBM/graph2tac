from pathlib import Path
import sys
import yaml 

from build_model_spec import create_and_push_spec

def main():
    benchmark_params_yaml = Path(sys.argv[1])
    
    with benchmark_params_yaml.open() as f:
        params_dict = yaml.safe_load(f)
    
    for benchmk_params in params_dict["benchmark"]["benchmarks"]:
        spec_commit = create_and_push_spec(
            git_repo=benchmk_params["spec_repo"],
            git_branch=benchmk_params["spec_branch"],
            params=benchmk_params["spec"]
        )
        print(benchmk_params["spec_repo"], benchmk_params["spec_branch"], spec_commit)

if __name__ == "__main__":
    main()