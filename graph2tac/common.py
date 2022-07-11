import os, glob
import capnp
import hashlib
import logging
import sys
from pathlib import Path
"""

load_capnp(data_dir: str)

returns python capnp api loaded from a unique supported .capnp file found
in the target roodir and exits on error (not unique .capnp file or unsupported Magic Number)

Example:

data_dir = "<path graph2tac>/tests/unit-test-propositional/dataset"
api = load_capnp(data_dir)

I suggest to get rid of specifying location of dataset and capnp file
anywhere in the code and make the capnp api to be loaded only as a function as a data_dir
which is passed as an argument to the run scripts (Vasily)

"""
def load_capnp(data_dir: str):
    MAGIC_LABELLED_GRAPH_API_V3 = "@0xafda4797418def92;"
    """
    returns loaded capnp api of a single capnp file from data_dir
    """
    capnp_filenames = glob.glob(os.path.join(data_dir, '*.capnp'))
    if len(capnp_filenames) != 1:
        raise Exception(f"Error: expected to find unique capnp file in {data_dir},"
                        f"but instead found {capnp_filenames}")

    capnp_filename = capnp_filenames[0]
    with open(capnp_filename) as f:
        magic = f.readline().strip()
        if magic != MAGIC_LABELLED_GRAPH_API_V3:
            raise Exception(f"Error in loading {capnp_filename}:"
                            f"unknown magic number of capnp protocol {magic}")
        else:
            capnp_api = capnp.load(capnp_filename)
            print(f"LOADING | loaded capnp: {capnp_filename}")
            return capnp_api


def uuid(data_dir: Path):
    m = hashlib.md5()
    m.update(os.fsencode(data_dir.expanduser().resolve()))
    return m.hexdigest()[:6]

logger = logging.getLogger('graph2tac')
logger.verbose = lambda message: logger.log(level=15, msg=message)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(name)s:%(levelname)s - %(message)s'))
logger.addHandler(handler)
