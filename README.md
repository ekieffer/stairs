# STAIRS-code

![Python Logo](https://www.python.org/static/community_logos/python-logo.png "Sample inline image")

This repository contains the source code of the STAIRS project. The University of Luxembourg and the European Investment Bank (EIB) through the STAREBEI programme are working together to encourage private equity partners to invest in innovative and sustainable technologies. The funded research project “Sustainable and Trustworthy Artificial Intelligence Recommitment System (STAIRS) ” proposes an innovative approach to generate efficient recommitment strategies to guide institutional investors with the aid of AI-based algorithms.

## Installation

We strongly recommend to create a virtual environment to avoid any interaction with your system python. Please note that only Python 3.6+ are supported.

### Dependencies

```bash  
$> sudo apt install build-essential git
$> python3 -m venv stairs-env
$> source stairs-env/bin/activate
$(stairs-env)> git clone git@github.com:ekieffer/stairs-code.git
$(stairs-env)> cd stairs-env
$(stairs-env)> python3 -m pip install -r requirements.txt
$(stairs-env)> cd stairs; python setup.py build_ext --inplace; cd ..
```

## Generate ESG scores

* Create ESG score with specific skewness and correlation

```bash
python -m stairs.esg_score --plot --save ./data/lib_cashflows_esg.h5  --strategy proba -a 50 -c 0.9  ./data/lib_cashflows.h5
```

## Generate initial portfolios

* Create initial portfolios with overcommitment for training and validation

```bash
python -m stairs.setup_portfolios -p 1000 -o 0 ./data/lib_cashflows_esg.h5 ./data/training_portfolios.h5
python -m stairs.setup_portfolios -p 250  -o 0.3 ./data/lib_cashflows_esg.h5 ./data/validation_portfolios.h5
```

## Start Parallel training


* Start Parallel training with config file

```bash
nohup ipcluster start -n 10 &
python -m stairs.experiment.py configs/example.cfg
ipcluster stop
```

## Start validation

* Test the heuristics obtained from parallel training

```bash
python -m stairs.validation configs/example.cfg
```

## Generate statistics

* Generate statistics and plots from portfolios obtained after validation

```bash
python -m stairs.statistics /tmp/dir_portfolios/portfolios-heuristic0.h5 /tmp/dir_portfolios/portfolios-heuristic1.h5 /tmp/dir_portfolios/portfolios-heuristic2.h5
```

