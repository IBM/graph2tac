from pathlib import Path
import json
import yaml

for json_file in Path("tests/data").glob("**/expected*.json"):
    with json_file.open() as f:
        d = json.load(f)
    with json_file.with_suffix('.yaml').open("w") as f:
        yaml.safe_dump(d, f)
    json_file.unlink()  # delete