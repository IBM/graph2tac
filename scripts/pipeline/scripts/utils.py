import os
from pathlib import Path
import yaml

class Utils:
    @staticmethod
    def setup_conda(conda_env_path: Path, conda_params: dict):
        print()
        print("========")
        print("Setting up conda env")
        print("========")
        print()
        os.system(f"conda create -y -p {conda_env_path} python={conda_params['python']}")
        for cmd in conda_params["setup_cmds"]:
            os.system(f"conda run -p {conda_env_path} {cmd}") 
        assert conda_env_path.exists()

    @staticmethod
    def setup_opam(conda_run_prefix: str, opam_switch_path: Path, opam_params: dict):
        print()
        print("========")
        print("Setting up opam switch")
        print("========")
        print()
        os.system(f"{conda_run_prefix} opam switch create {opam_switch_path} --empty")

        # to run opam commands, make a temporary script and run that
        install_script_sh = opam_switch_path / "setup_opam.sh"
        script = [f"eval $(opam env --switch={opam_switch_path})"]
        for cmd in opam_params["setup_cmds"]:
            script.append(cmd)
        install_script_sh.write_text("\n".join(script))

        os.system(f"{conda_run_prefix} sh {install_script_sh}")
        assert (opam_switch_path / "_opam").exists()

    @staticmethod
    def get_run_dir_name():
        run_dir_name = str(Path(".").resolve().parent.name)  # parent directory
        assert run_dir_name.startswith("20"), (
            f"Not run from expected directory structure.\n"
            f"wkdir: {Path('.').resolve()}\n"
            f"Unexpected parent:{run_dir_name}"
        )
        return run_dir_name

    @staticmethod
    def get_params():
        with Path("../params.yaml").open() as f:
            params = yaml.safe_load(f)
        return params

    @staticmethod
    def get_directories() -> dict[str, Path]:
        directories = {}
        directories["logs"] = Path("../logs").resolve()
        directories["scripts"] = Path("../scripts").resolve()
        directories["params"] = Path("../params").resolve()
        directories["results"] = Path("../results").resolve()
        directories["workdir"] = Path("../workdir").resolve()

        for key, path in directories.items():
            assert path.exists(), f"{key} directory doesn't exist: {path}"
        
        return directories