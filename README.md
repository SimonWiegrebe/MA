# What next? Modeling human behavior using smartphone usage data and (deep) recommender systems
This repository only contains pre- and post-processing code, result tables, figures, the report and the presentation. It does not contain the code architecture and preprocessed data for running algorithms, nor raw results.

## Folder Structure
Please unzip "final". It contains three folders, "data", "MA" and "session-rec".

final
    ├── data         # Contains raw data files
    ├── MA           # Equivalent to this GitHub repository
    ├── session-rec  # Contains code architecture and preprocessed data for running algorithms as well as raw results; based on https://github.com/rn5l/session-rec

## Running Algorithms
For running algorithms, only the folder "session-rec" is required. To do so:

- Download and install Anaconda or Miniconda
- In a shell, navigate to the folder called "session-rec"
- From the command line run:
  - conda env create --file environment_cpu.yml
  - conda activate srec37 (to activate the conda environment)
  - conda uninstall pytables (because it seems to install, but throws an error afterwards)
  - conda install -c conda-forge pytables
  - pip install libpython
  - pip install mkl toolbox
