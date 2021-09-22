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




def recorder(path,heuristics):
    """
    Record all heuristics to a txt file
    """
    with open(path, 'w') as opened_file:
        for h in heuristics:
            opened_file.write(str(h)+"\n")


def read_heuristics(path):
    heuristics=[]
    with open(path,"r") as fd:
     for k,line in enumerate(fd.readlines()):
         line=line.replace("\n","")
         heuristics.append(line)
    return heuristics


def main(client,**kwargs):
    if kwargs["distributed"]:
        ds = Distributed_Simulation(client,kwargs["cashflows"],kwargs["portfolios"],kwargs["cashflows_freq"],kwargs["commitments_iters"],kwargs["funds_per_recommitment"],kwargs["with_esg"])
    else:
        ds = Simulation(kwargs["cashflows"],kwargs["portfolios"],kwargs["cashflows_freq"],kwargs["commitments_iters"],kwargs["funds_per_recommitment"],kwargs["with_esg"])
    pset = EvolvableHeuristic.get_functions_set()
    model = GPLearner(ds, pset, kwargs["pop"], kwargs["crossover"], kwargs["mutation"], kwargs["gen"],kwargs["tf_logsdir"])
    start = time.time()
    return model.learn(warm_start=kwargs.get("warm_start",[]))


if __name__ == "__main__":
    kwargs = {}
    # Defaults
    ## Simulation
    kwargs["distributed"] = 1
    kwargs["cashflows_freq"] = 4
    kwargs["commitments_iters"] = 104
    kwargs["funds_per_recommitment"] = 1
    kwargs["with_esg"] = 1
    ## Evolution
    kwargs["pop"] = 100
    kwargs["gen"] = 100
    kwargs["crossover"] = 0.85
    kwargs["mutation"] = 0.1

    assert  len(sys.argv) == 2, "This script only takes a configuration file as input"
    config = configparser.ConfigParser()
    assert os.path.exists(sys.argv[1]),"Invalid config file"
    config.read(sys.argv[1])
    sections = config.sections()
    # Read section Simulation
    assert "SIMULATION" in sections, "No simulation section in the config file"
    simu_section = config["SIMULATION"]
    kwargs["distributed"] = bool(int(simu_section.get("distributed",kwargs["distributed"])))
    kwargs["cashflows"] = simu_section.get("cashflows",None)
    assert kwargs["cashflows"] is not None,"No cashflows Library provided"
    kwargs["portfolios"] = simu_section.get("training_portfolios",None)
    assert kwargs["portfolios"] is not None,"No initial portfolios provided"
    kwargs["cashflows_freq"] = int(simu_section.get("cashflows_freq",kwargs["cashflows_freq"]))
    kwargs["commitments_iters"] = int(simu_section.get("commitments_iters",kwargs["commitments_iters"]))
    kwargs["funds_per_recommitment"] = int(simu_section.get("funds_per_recommitment",kwargs["commitments_iters"]))
    kwargs["with_esg"] = bool(int(simu_section.get("with_esg",kwargs["with_esg"])))
    # Read section Evolution
    assert "EVOLUTION" in sections, "No evolution section in the config file"
    evo_section = config["EVOLUTION"]
    kwargs["pop"] = int(evo_section.get("pop",kwargs["pop"]))
    kwargs["gen"] = int(evo_section.get("gen",kwargs["gen"]))
    kwargs["crossover"] = float(evo_section.get("crossover",kwargs["crossover"]))
    kwargs["mutation"] = float(evo_section.get("mutation",kwargs["mutation"]))
    path_to_h = evo_section.get("warm_start",None)
    if path_to_h is not None:
        assert os.path.exists(path_to_h), "Heuristic file for warm start is invalid"
        kwargs["warm_start"] = read_heuristics(path_to_h)
    # Read section output
    assert "OUTPUT" in sections, "No evolution section in the config file"
    output_section = config["OUTPUT"]
    kwargs["heuristics"] = output_section.get("heuristics",None)
    assert kwargs["heuristics"] is not None,"No output path provided"
    kwargs["tf_logsdir"] = output_section.get("tf_logsdir",None)
    # Now start the evolution 
    rc = None
    if kwargs["distributed"]:
        if os.environ.get('IPY_PROFILE',None) is not None:
            rc=ipp.Client(profile=os.environ['IPY_PROFILE'])
        else:
            rc=ipp.Client()
    pop,log,hof = main(rc,**kwargs)
    recorder(kwargs["heuristics"], hof)
    data=dict(pop=pop,hof=hof,log=log)
    pickle.dump(data,open("{0}.gp".format(kwargs["heuristics"]),"wb"))




    



