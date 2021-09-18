# What next? Modeling human behavior using smartphone usage data and (deep) recommender systems
This repository only contains pre- and post-processing code, result tables, figures, the report and the presentation. It does not contain the code architecture or preprocessed data for running algorithms, nor raw results.

## Folder Structure
Please unzip the zip-file "final". It contains three folders: ``data``, ``MA`` and ``session-rec``.

- ``data``:         Contains raw data files
- ``MA``:           Equivalent to this GitHub repository
- ``session-rec``:  Contains code architecture and preprocessed data for running algorithms as well as raw results; based on https://github.com/rn5l/session-rec

## Run Algorithms
In order to run algorithms, only the folder ``session-rec`` is required.

### Setup
- Download and install Anaconda or Miniconda
- In a shell, navigate to the folder called "session-rec"
- From the command line, run:
  - ``conda env create --file environment_cpu.yml``
  - ``conda activate srec37`` (to activate the conda environment)
  - ``conda uninstall pytables`` (because it seems to install, but throws an error afterwards)
  - ``conda install -c conda-forge pytables``
  - ``pip install libpython``
  - ``pip install mkl toolbox``

### Run Configuration
- In a shell, navigate to the folder called ``session-rec``
- Copy the desired configuration file(s) into the folder ``conf/in``
  - all configuration files can be found in the folder called ``conf``
  - e.g., ``conf/testing/app-level/multiple/vsknn_EBR-window_1.yml`` for the fully tuned VSKNN_EBR algorithm to be run on window 1 app-level test data
- from the command line, run: ``python run_config.py conf/in conf/out``
