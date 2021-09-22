from stairs.libraries import Library
from stairs.utils import simcorr, to_proba_scores, skew_random, binary_tournament
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import numpy as np
import argparse
import sys
import seaborn
import pandas



def simulation_selection(strategy,tvpis,esg_values,n=1500):
    if strategy == "uniform":
        sample_indexes = np.random.choice(len(esg_values),n)
    elif strategy == "bt":
       sample_indexes = [binary_tournament(esg_values) for _ in range(n)]
    else:
        sample_indexes = np.random.choice(len(esg_values),n,p=to_proba_scores(esg_values))
    selected = ["not selected"]*len(esg_values) 
    for i in sample_indexes:
        selected[i]="selected"
    return selected


def main(**kwargs):
    lib = Library()
    lib.load_library(kwargs["path_lib_cashflows"])
    TVPIS=[]
    # Retrieve a vector of TVPIS
    for i in range(len(lib.funds_set)):
        fund = lib.funds_set[i]
        TVPIS.append(fund.get_param('TVPI'))
    esg_values = simcorr(TVPIS,kwargs["alpha"],kwargs["correlation"])
    esg_values = esg_values + np.abs(np.min(esg_values)) + 1 
    print("Correlation obained: {0}".format(np.corrcoef(TVPIS,esg_values)[0,1]))
    if kwargs["save"] is not None:
        for i in range(len(lib.funds_set)):
            fund = lib.funds_set[i]
            fund.set_param("ESG",esg_values[i])
        lib.set_annotation("esg_alpha_skewness",kwargs["alpha"])
        lib.set_annotation("esg_tvpi_correlation",kwargs["correlation"])
        lib.save(kwargs["save"])
    if kwargs["plot"]:
        cat = simulation_selection(kwargs["strategy"],TVPIS,esg_values)
        tips = pandas.DataFrame({"tvpis": TVPIS, "ESG scores": esg_values,"cashflows":cat})
        sns = seaborn.jointplot(x="tvpis", y="ESG scores",hue="cashflows", data=tips)
        sns.plot_joint(seaborn.kdeplot, n_levels=3)
        sns.fig.suptitle("Correlation between TVPIs and ESG scores")
        sns.fig.subplots_adjust(top=0.95) # Reduce plot to make room 
        plt.show()




        


if __name__ == "__main__":
    kwargs = dict()
    kwargs["path_lib_cashflows"] = None
    kwargs["correlation"] = 0.5
    kwargs["alpha"] = 0
    kwargs["plot"] = False
    kwargs["save"] = None
    kwargs["strategy"] = "uniform"
	
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--correlation', type=float or sys.exit("Need a floating value between 0 and 1."), default=kwargs["correlation"], help="Correlation with funds TVPIs")
    parser.add_argument('-a', '--alpha', type=float, default=kwargs["alpha"], help="Skewness of the ESG scores")
    parser.add_argument('-s', '--strategy',nargs='?',default="uniform",choices=["uniform","bt","proba"], help="Sampling simulation strategy (only available with --plot)")
    parser.add_argument('--save', dest='save', default=None,help="Save new libraries with ESG values")
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.add_argument('path_lib_cashflows')
    args = parser.parse_args()
    kwargs["path_lib_cashflows"] = args.path_lib_cashflows
    kwargs["correlation"] = float(args.correlation)
    kwargs["alpha"] = float(args.alpha)
    kwargs["plot"] = bool(args.plot)
    kwargs["save"] = args.save
    kwargs["strategy"] = args.strategy
    print(kwargs)
    main(**kwargs)



    





