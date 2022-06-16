import argparse
import os
from graph2tac.loader.data_server import find_bin_files
from pathlib import Path
from graph2tac.loader.clib.loader import capnp_unpack

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    args = parser.parse_args()
    target_dir = Path(args.data_dir).expanduser().absolute()

    print(f'unpacking all bin files {target_dir}')

    for fname in find_bin_files(target_dir):
        capnp_unpack(os.fsencode(fname))


if __name__ == '__main__':
    main()
