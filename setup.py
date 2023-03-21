from setuptools import find_packages, setup

# please indent the lists with one element per line
# for convenience of version control tools

setup(
    name='graph2tac',
    packages=find_packages(),  # find all packages in the project instead of listing them 1-by-1
    version='0.1.0',
    description='graph2tac converts graphs to actions',
    author=' Mirek Olsak, Vasily Pestun, Jason Rute, Fidel I. Schaposnik Massolo',
    python_requires='>=3.9',
    include_package_data=True,
    entry_points={'console_scripts':
                  [
                      'g2t-train-tfgnn=graph2tac.tfgnn.train:main',
                      'g2t-server=graph2tac.loader.predict_server:main',
                      'g2t-train-hmodel=graph2tac.loader.hmodel:main',
                      'g2t-tfgnn-predict-graphs=graph2tac.tfgnn.plot_predict_graphs:main',
                  ]},
    license='MIT',
    install_requires=[
        'keras>=2.8',
        'tensorflow>=2.9.0,<2.10', # TODO(jrute): Try to upgrade to >=2.10
        'tensorflow_gnn>=0.2.0,<0.3',  # TODO(jrute): Try to upgrade to >=0.3
        'tqdm',
        'numpy',
        'pycapnp',
        'psutil',
        'pyyaml',
        'graphviz',
        'pytactician @ git+https://git@github.com/coq-tactician/coq-tactician-reinforce@a2f7f68a4c9321dba9d65dfc986eed97e939b7da',
        'pytest',
    ]
)
