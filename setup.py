from pathlib import Path
import sys
import setuptools
from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext
import subprocess

# These next three lines fetch and import numpy, which is needed for installation
import setuptools.dist
setuptools.dist.Distribution().fetch_build_eggs(['Cython>=0.15.1', 'numpy>=1.10'])
import numpy


# please indent the lists with one element per line
# for convenience of version control tools

CAPNP_V=11


CLIB_DIR = Path("./graph2tac/loader/clib")
CAPNP_FILE = CLIB_DIR / f"graph_api_v{CAPNP_V}.capnp"

class BuildCapnpFiles(build_ext):
    """
    Build CAPNP library before doing other build extensions

    This code will be run when using
      pip install <this project>
    or
      pip install -e <this project>

    Use -vvv arguement with pip install to display print statements in this class.

    For this to work, must have cmdclass={'build_ext': BuildCapnpFiles} in setup below.
    """

    def build_extensions(self):
        build_ext.build_extensions(self)

    def run(self):
        print("Compiling .capnp files")
        subprocess.check_output(["capnpc", "-oc++", str(CAPNP_FILE)])
        # rename .capnp.c++ to .capnp.cpp
        CAPNP_FILE.with_suffix(".capnp.c++").rename(CAPNP_FILE.with_suffix(".capnp.cpp"))
        build_ext.run(self)



compile_mac_options =  ["-mmacosx-version-min=10.15"] if sys.platform == 'darwin' else []
my_c_module = setuptools.Extension('graph2tac.loader.clib.loader',
                                   sources=[
                                       f'graph2tac/loader/clib/graph_api_v{CAPNP_V}.capnp.cpp',
                                       'graph2tac/loader/clib/loader.cpp',
                                   ],
                                   include_dirs=[numpy.get_include()],
                                   extra_compile_args=(
                                       [
                                       "-std=c++17",
                                       "-O3",
                                       "-I/opt/local/include",
                                       f"-DCAPNP_V={CAPNP_V}"
                                        ] +
                                       compile_mac_options
                                       ),
                                   extra_link_args=[
                                       "-lkj",
                                       "-lcapnp",
                                       "-lcapnp-rpc",
#                                       "-lstdc++fs",
#                                      "-ltbb",         # we may need threads library but ok without it in current code
                                       "-L/opt/local/lib",
                                       ]
                                   )

# the reason for the static linking to stdc++fs library is
# a necessety to run this software on a legacy linux system that misses GLIBCXX_3.4.26
# very helpful explanation was given at
# https://stackoverflow.com/questions/63902528/program-crashes-when-filesystempath-is-destroyed


setup(
    name='graph2tac',
    packages=find_packages(),  # find all packages in the project instead of listing them 1-by-1
    cmdclass={'build_ext': BuildCapnpFiles},
    ext_modules=[my_c_module],
    version='0.1.0',
    description='graph2tac converts graphs to actions',
    author=' Mirek Olsak, Vasily Pestun, Jason Rute, Fidel I. Schaposnik Massolo',
    python_requires='>=3.9',
    include_package_data=True,
    package_data={'graph2tac.loader': [f'clib/graph_api_v{CAPNP_V}.capnp']},
    entry_points={'console_scripts':
                  [
                      'g2t-preprocess=graph2tac.loader.preprocess:main',
                      'g2t-train=graph2tac.tf2.train:main',
                      'g2t-stat=graph2tac.loader.graph_stat:main',
                      'g2t-printlog=graph2tac.printlog:main',
                      'g2t-unpack=graph2tac.loader.unpack:main',
                      'g2t-server=graph2tac.loader.predict_server:main',
                      'g2t-train-hmodel=graph2tac.loader.hmodel:main'
                  ]},
    license='MIT',
    install_requires=[
        'keras>=2.8',
        'tensorflow>=2.8',
        'fire',
        'pycapnp',
        'psutil',
        'tensor2tensor',
        'dataclasses-json',
        'pyyaml',
        'graphviz'
    ]
)
