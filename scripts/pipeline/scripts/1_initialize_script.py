import os
from pathlib import Path
import yaml

def setup_run_directory():
    print("Creating directories...")
    Path("../logs").mkdir(exist_ok=True)
    Path("../params").mkdir(exist_ok=True)
    Path("../workdir").mkdir(exist_ok=True)
    Path("../results").mkdir(exist_ok=True)

def main():
    with Path("../params.yaml").open() as f:
        params = yaml.safe_load(f)
    
    # set up the directories
    setup_run_directory()

if __name__ == "__main__":
    main()