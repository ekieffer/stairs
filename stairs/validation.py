import os
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"]="0"
import ipyparallel as ipp
from stairs.heuristicLearners import GPLearner
from stairs.simulation import Simulation
from stairs.distributed import Distributed_Simulation
from stairs.libraries import Library
from stairs.heuristics import EvolvableHeuristic
from deap import gp
import time
import numpy as np
import configparser
import sys
import pickle
import pathlib
import sympy

LATEX=True

## Should be placed in a new module -- only for testing here
def sympy_sub(a,b):
    return sympy.Add(a,-b)

def sympy_div(a,b):
    return a * sympy.Pow(b, -1)

def sympy_opposite(a):
    return sympy.Mul(a, -1)

def sympy_inverse(a):
    return sympy.Pow(a, -1)

SYMPY_NAMESPACE={'add': sympy.Add,
                 'sub': sympy_sub,
                 'mul': sympy.Mul,
                 'protectedDiv': sympy_div,
                 'inverse': sympy_inverse,
                 'opposite': sympy_opposite}

def add_symbol_to_namespace():
    pset = EvolvableHeuristic.get_functions_set()
    for arg in pset.arguments:
        SYMPY_NAMESPACE[arg] = sympy.Symbol(args)

def sympify_and_latex(heuristic_str):
    return sympy.latex(sympy.sympify(heuristic_str,locals=SYMPY_NAMESPACE))
        
######################################################################


def read_heuristics(path):
    heuristics=[]
    with open(path,"r") as fd:
     for k,line in enumerate(fd.readlines()):
         line=line.replace("\n","")
         heuristics.append(line)
    return heuristics




def main(**kwargs):
    ds = Simulation(kwargs["cashflows"],kwargs["portfolios"],kwargs["cashflows_freq"],kwargs["commitments_iters"],kwargs["funds_per_recommitment"],kwargs["with_esg"])
    pset = EvolvableHeuristic.get_functions_set()
    for k,heuristic in enumerate(kwargs["heuristics"]):
        func = gp.compile(gp.PrimitiveTree.from_string(heuristic,pset),pset)
        ds.execute(EvolvableHeuristic(func))
        lib_new = Library()
        for portfolio in ds.portfolios:
            lib_new.append(portfolio)
        dir_res = pathlib.Path(kwargs["results"])
        if not dir_res.exists():
            dir_res.mkdir(parents=True)
        file_res = dir_res / "portfolios-heuristic{0}.h5".format(k) 
        if LATEX:
            lib_new.set_annotation("heuristic",sympify_and_latex(heuristic))
        else:
            lib_new.set_annotation("heuristic",heuristic)
        lib_new.save(str(file_res))


if __name__ == "__main__":
    kwargs = {}
    # Defaults
    ## Simulation
    kwargs["distributed"] = 0
    kwargs["cashflows_freq"] = 4
    kwargs["commitments_iters"] = 104
    kwargs["funds_per_recommitment"] = 1
    kwargs["with_esg"] = 1

    assert  len(sys.argv) == 2, "This script only takes a configuration file as input"
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    sections = config.sections()
    # Read section Simulation
    assert "SIMULATION" in sections, "No simulation section in the config file"
    simu_section = config["SIMULATION"]
    kwargs["distributed"] = bool(int(simu_section.get("distributed",kwargs["distributed"])))
    kwargs["cashflows"] = simu_section.get("cashflows",None)
    assert kwargs["cashflows"] is not None,"No cashflows Library provided"
    kwargs["portfolios"] = simu_section.get("validation_portfolios",None)
    assert kwargs["portfolios"] is not None,"No initial portfolios provided"
    kwargs["cashflows_freq"] = int(simu_section.get("cashflows_freq",kwargs["cashflows_freq"]))
    kwargs["commitments_iters"] = int(simu_section.get("commitments_iters",kwargs["commitments_iters"]))
    kwargs["funds_per_recommitment"] = int(simu_section.get("funds_per_recommitment",kwargs["commitments_iters"]))
    kwargs["with_esg"] = bool(int(simu_section.get("with_esg",kwargs["with_esg"])))
    # Read section output
    assert "OUTPUT" in sections, "No evolution section in the config file"
    output_section = config["OUTPUT"]
    kwargs["heuristics"] = output_section.get("heuristics",None)
    assert kwargs["heuristics"] is not None,"No path provided"
    kwargs["heuristics"] = read_heuristics(kwargs["heuristics"])
    kwargs["results"] = output_section.get("results",None)
    assert kwargs["heuristics"] is not None,"No path provided"
    # Now start the validation 
    main(**kwargs)
