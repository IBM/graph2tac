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
        'tensorflow>=2.12.0,<2.13',
        'tensorflow_gnn>=0.2.0,<0.3',  # TODO(jrute): Try to upgrade to >=0.3
        'protobuf<4.0',
        'tqdm',
        'numpy',
        'pycapnp',
        'psutil',
        'pyyaml',
        'graphviz',
        'pytactician==15.1',
        'pytest',
    ]
)
