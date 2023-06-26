#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import tempfile
from typing import Optional, List, Dict

import yaml


OPAM_PATH = Path("coq-graph2tac.opam")
OPAM_TEMPLATE = """opam-version: "2.0"
synopsis: "Graph neural network that predicts tactics for Tactician"
description: ""
maintainer: ["Lasse Blaauwbroek <lasse@blaauwbroek.eu>"]
authors: [
  "Lasse Blaauwbroek"
  "Mirek Olsak"
  "Vasily Pestun"
  "Jelle Piepenbrock"
  "Jason Rute"
  "Fidel I. Schaposnik Massolo"
]
homepage: "https://coq-tactician.github.io"
bug-reports:
  "https://github.com/pestun/graph2tac/issues"
build: [
  [ "tar" "-xzf" "model.tar.gz" "--one-top-level" "--strip-components" "1" ]
]
install: [
  [ "mkdir" "-p" "%{share}%/%{name}%/" ]
  [ "cp" "-r" "model/" "%{share}%/%{name}%/" ]

  [ "cp" "Graph2TacConfig.v" "%{lib}%/coq/user-contrib/Tactician/Graph2TacConfig.v" ]

  # We have to make sure that our injection flags get loaded after the injection flags of coq-tactician-reinforce.
  # We do this by using a name that is guaranteed to sort after coq-tactician-reinforce.
  [ "mkdir" "-p" "%{share}%/coq-tactician/plugins/coq-tactician-reinforce-%{name}%/" ]
  [ "cp" "injection-flags" "%{share}%/coq-tactician/plugins/coq-tactician-reinforce-%{name}%/" ]
]
dev-repo: "git+https://github.com/pestun/graph2tac.git"
pin-depends: [
  [
    "coq-tactician.8.11.dev"
    "git+https://github.com/coq-tactician/coq-tactician.git#COQ_TACTICIAN_COMMIT"
  ]
  [
    "coq-tactician-reinforce.8.11.dev"
    "git+ssh://git@github.com/coq-tactician/coq-tactician-reinforce.git#COQ_TACTICIAN_REINFORCE_COMMIT"
  ]
]
depends: [
  "coq-tactician-reinforce"
  "coq-tactician"
]
extra-source "model.tar.gz" {
  src: "MODEL_SOURCE_URL"
  checksum: "md5=MODEL_MD5SUM"
}
substs: [
  "Graph2TacConfig.v"
]
"""
OPAM_PARAMS = ["COQ_TACTICIAN_COMMIT", "COQ_TACTICIAN_REINFORCE_COMMIT", "MODEL_SOURCE_URL", "MODEL_MD5SUM"]

PREREQUISITES_PATH = Path("prerequisites")
PREREQUISITES_TEMPLATE = """#!/usr/bin/env bash
set -ue
python3 -m venv ./venv
. ./venv/bin/activate
pip install GRAPH2TAC_PATH
"""
PREREQUISITES_PARAMS = ["GRAPH2TAC_PATH"]

INJECTION_FLAGS_PATH = Path("injection-flags")
INJECTION_FLAGS_TEMPLATE = """-l Graph2TacConfig.v
"""
INJECTION_FLAGS_PARAMS = []

IN_V_PATH = Path("Graph2TacConfig.v.in")
IN_V_TEMPLATE = """INJECTIONS
"""
IN_V_PARAMS = ["INJECTIONS"]

UPDATE_ENV_PATH = Path("update-env")
UPDATE_ENV_TEMPLATE = """#!/usr/bin/env bash
comm -13 <(env | sort)  <(source ./venv/bin/activate && env | sort)
"""
UPDATE_ENV_PARAMS = []

def build_file_from_template(
    filepath: Path, 
    template: str, 
    parameter_variables: List[str], 
    parameter_values: List[str]
):
    """Build a file from a template.

    :param filepath: Path where the file will be created.
    :type filepath: Path
    :param template: A multi-line template including keywords to be replaced.
    :type template: str
    :param parameter_variables: List of keywords to replace in the template.
    :type parameter_variables: List[str]
    :param parameter_values: List of values to replace in each keyword.
    :type parameter_values: List[str]
    """
    file_parts: List[str] = []
    template_remainder = template
    assert len(parameter_variables) == len(parameter_values)
    assert all(isinstance(v, str) for v in parameter_values), f"Expected a list of strings.  Got: {parameter_values}"
    for p_var, p_value in zip(parameter_variables, parameter_values):
        assert template_remainder.count(p_var) == 1
        part, template_remainder = template_remainder.split(p_var)
        file_parts.append(part)
        file_parts.append(p_value)
    file_parts.append(template_remainder)
    with filepath.open("w") as f:
        f.write("".join(file_parts))

def read_params() -> tuple[str, dict]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--git-branch",
        type=str,
        help="Git branch to save this to. (For now assume it is unique.)"
    )
    parser.add_argument(
        "--params",
        type=Path,
        help="YAML parameter file"
    )
    args = parser.parse_args()

    # check inputs
    assert args.params.is_file(), f"{args.params} must be an existing file"
    assert str(args.params).endswith(".yaml"), f"{args.params} must be a .yaml file"
    
    with args.params.open() as f:
        param_dict=yaml.safe_load(f)
      
    return args.git_branch, param_dict

def clone_repo(temp_dir: Path, branch_or_commit: str, new_branch: Optional[str]):
    os.chdir(temp_dir)
    os.system("git clone git@github.com:coq-tactician/coq-graph2tac-trained.git")
    #TODO: check that it is there
    os.chdir("coq-graph2tac-trained")
    os.system(f"git checkout {branch_or_commit}")
    #TODO: check that this worked
    os.system(f"git checkout {new_branch} || git checkout -b {new_branch}")
    #TODO: check that this worked

def modify_files(temp_dir: Path, params: Dict[str, str]):
    repo = temp_dir / "coq-graph2tac-trained"
    os.chdir(repo)
    build_file_from_template(
        filepath=OPAM_PATH,
        template=OPAM_TEMPLATE,
        parameter_variables=OPAM_PARAMS,
        parameter_values=[
          params["coq_tactician_commit"], 
          params["coq_tactician_reinforce_commit"],
          params["model_source_url"],
          params["model_md5sum"],
        ]
    )
    build_file_from_template(
        filepath=PREREQUISITES_PATH,
        template=PREREQUISITES_TEMPLATE,
        parameter_variables=PREREQUISITES_PARAMS,
        parameter_values=[
            params["graph2tac_path"],
        ]
    )
    build_file_from_template(
        filepath=INJECTION_FLAGS_PATH,
        template=INJECTION_FLAGS_TEMPLATE,
        parameter_variables=INJECTION_FLAGS_PARAMS,
        parameter_values=[]
    )
    build_file_from_template(
        filepath=IN_V_PATH,
        template=IN_V_TEMPLATE,
        parameter_variables=IN_V_PARAMS,
        parameter_values=[
            "\n".join(params["injection_lines"]),
        ]
    )
    build_file_from_template(
        filepath=UPDATE_ENV_PATH,
        template=UPDATE_ENV_TEMPLATE,
        parameter_variables=UPDATE_ENV_PARAMS,
        parameter_values=[]
    )
    with Path("params.yaml").open("w") as f:
        yaml.safe_dump(params, f)

def commit_and_push_repo(temp_dir: Path, commit_message: str, git_branch: str) -> str:
    repo = temp_dir / "coq-graph2tac-trained"
    os.chdir(repo)
    print(os.popen("git diff").read())
    os.system("git add *")
    os.system(f"git commit --allow-empty-message --allow-empty -m '{commit_message}'")

    # push
    os.system(f"git push || git push --set-upstream origin {git_branch}")
    git_commit = os.popen('git rev-parse HEAD').read().strip()
    return git_commit

def communicate_git_commit(git_commit: str):
    print(f"Git commit: {git_commit}")

def create_and_push_spec(git_branch: str, params: Dict[str, str]) -> str:
    assert set(params.keys()) == {
        "coq_tactician_commit",
        "coq_tactician_reinforce_commit",
        "model_source_url",
        "model_md5sum",
        "graph2tac_path",
        "injection_lines",
        "description"
    }, params.keys()

    cwd = Path.cwd()

    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_dir = Path(tmpdirname)
        clone_repo(
            temp_dir=temp_dir,
            branch_or_commit = "main",
            new_branch = git_branch
        )
        modify_files(
            temp_dir=temp_dir,
            params=params
        )
        git_commit = commit_and_push_repo(
            temp_dir=temp_dir,
            commit_message=params["description"],
            git_branch=git_branch
        )

    os.chdir(cwd)
    return git_commit

def main():
    git_branch, params = read_params()
    git_commit = create_and_push_spec(git_branch=git_branch, params=params)
    communicate_git_commit(git_commit)

if __name__ == "__main__":
    main()
    