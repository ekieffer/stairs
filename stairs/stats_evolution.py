import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse
import logging
import glob
import os
import pickle
import re
from scipy.stats import mstats
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
creator.create("FitnessMinGP", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMinGP)


sns.set_theme(style="darkgrid")
np.seterr(divide='ignore', invalid='ignore')
logging.basicConfig(level=logging.INFO,format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
MAX_COLS=2

plt.rcParams.update({"text.usetex": True,'axes.titlesize':'x-large'})
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


plt.rcParams.update({"text.usetex": True,'axes.titlesize':38})
plt.rcParams.update({'font.size': 22,'legend.fontsize': 28,
         'axes.labelsize': 'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'})



class EVOSTATS:

    def __init__(self,heuristics_folder):
        logging.info("Start loading all runs")
        self.data = [ self.load_data(folder) for folder in heuristics_folder]
        logging.info("Done")
        logging.info("Start loading all corresponding heuristics")
        self.heuristics = [ self.load_heuristics(folder) for folder in heuristics_folder]
        logging.info("Done")
        logging.info("Generate DataFrames")
        self.df_fitness = [ self.build_df("fitness",self.data[i]) for i in range(len(self.data))] 
        self.df_size = [ self.build_df("size", self.data[i]) for i in range(len(self.data))]
        logging.info("Dataframes created")



    def load_data(self,heuristics_folder):
        lst = glob.glob(os.path.join(heuristics_folder,"*.gp"))
        data = dict()
        for log in lst:
            run = os.path.splitext(log)[0].split("_")[-1]
            data[run] = pickle.load(open(log,'rb'))
        return data


    def load_heuristics(self,folder):
        heuristics={}
        for it in os.scandir(folder):
            run = it.name.split("_")[-1]
            heuristics[run]=[]
            if re.match("heuristics_run[0-9]{0,2}$",it.name):
                with open(it.path,"r") as fd:
                    for k,line in enumerate(fd.readlines()):
                        line=line.replace("\n","")
                        heuristics[run].append(line)
        return heuristics


    def build_df(self,key,data):
        dct = dict()
        for run in data.keys():
            tab = data[run]["log"].chapters[key]
            if  dct.get("ngen",None) is None:
                dct["ngen"] = np.arange(len(tab.select("gen")))
            dct["avg_{0}".format(run)] =  np.array(tab.select("avg")).flatten()
            dct["min_{0}".format(run)] =  np.array(tab.select("min")).flatten()
            dct["max_{0}".format(run)] =  np.array(tab.select("max")).flatten()
            dct["std_{0}".format(run)] =  np.array(tab.select("std")).flatten()
            dct["evals_{0}".format(run)] =  np.array(tab.select("nevals")).flatten()
        df= pd.DataFrame(dct)
        df.set_index("ngen")
        return df


    def plot_mean_fitness(self,key,name):
        cm = plt.get_cmap('gist_rainbow')
        fig, ax = plt.subplots()
        for df in self.df_fitness:
            df1 = df.filter(regex=("{0}.*".format(key)))
            #df1.plot(title="Average Fitness",legend=False,ax=ax)
            #df1.mean(axis=1).plot(linewidth=5,label="Mean",legend=True,ax=ax)
            df1.mean(axis=1).plot(linewidth=5,label="Mean",legend=True,ax=ax)
        #ax.set_ylim(5,50)
        plt.xlabel("Generations")
        plt.title(name,fontsize=14)
        plt.show()

    def plot_mean_size(self,key,name,save=None):
        cm = plt.get_cmap('gist_rainbow')
        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 10.5)
        ax1 = ax.twinx()
        for df_fitness,df_size in zip(self.df_fitness,self.df_size):
            df1 = df_fitness.filter(regex=("{0}.*".format(key)))
            df2 = df_size.filter(regex=("{0}.*".format(key)))
            f=df1.mean(axis=1).plot(style="b-",linewidth=5,label="Average Objective value",ax=ax,backend="matplotlib")
            s=df2.mean(axis=1).plot(style=":",linewidth=5,label="Average function size",ax=ax1,color="red",backend="matplotlib")
        #ax.set_ylim(5,50)
        handles, labels = ax.get_legend_handles_labels()
        handles1, labels1 = ax1.get_legend_handles_labels()
        ax.legend(handles + handles1, labels + labels1, loc=5,frameon=False)
        plt.xlabel("Generations")
        plt.title(name,fontsize=14)
        plt.tight_layout()
        if save:
            plt.savefig(save,orientation="landscape",format="pdf")
        else:
            plt.show()


    def compute_runs_table(self):
        df_summary = pd.DataFrame()
        stats = []
        for k in range(len(self.data)):
           scores = []
           heuristics = []
           for i in range(len(self.data[k])):
               run = "run{0}".format(i+1)
               scores.append(self.data[k][run]["hof"][0].fitness.values[0])
               heuristics.append(str(self.data[k][run]["hof"][0]))
           stats.append(scores)    
           df_summary["xp{0}".format(k)] = scores
           df_summary["xp{0}_h".format(k)] = heuristics
        print(df_summary.to_excel("/tmp/training_comp.xlsx"))
        _,pval = mstats.kruskalwallis(*stats)
        if pval < 0.05:
            stacked_data = df_summary.filter(regex="xp[0-9]*$").stack().reset_index()
            stacked_data = stacked_data.rename(columns={'level_0': 'id',
                                            'level_1': 'xp',
                                            0:'objective value'})
            print(stacked_data)
            MultiComp = MultiComparison(stacked_data['objective value'],
                            stacked_data['xp'])

            # Show all pair-wise comparisons:
            # Print the comparisons
            with open('/tmp/training_tuckey.csv','w') as fh:
                fh.write(MultiComp.tukeyhsd().summary().as_csv())
            ax = sns.boxplot(x="xp", y="objective value", data=stacked_data, palette="Set3")
            plt.show()
        #### Check best and worse candidates
        best_and_worse = pd.DataFrame()
        for i in range(len(self.data)):
           best_and_worse["xp{0}".format(i)]=[df_summary[df_summary["xp{0}".format(i)] == df_summary["xp{0}".format(i)].min()]["xp{0}_h".format(i)].values[0],
                                              df_summary["xp{0}".format(i)].min(),df_summary["xp{0}".format(i)].idxmin()]     
        print(best_and_worse.transpose().rename(columns={0:"heuristic",1:"objective value",2:"index"}).to_markdown())

     





    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(nargs="+", dest="paths", help="One or more simulated portfolios in h5 format")
    parser.add_argument("-p","--plot",default=None, help="Output comparison table")
    args = parser.parse_args()
    print(args.paths)
    stats = EVOSTATS(args.paths)
    #stats.plot_mean_fitness("avg","Convergence curves")
    if not args.plot:
        stats.compute_runs_table()
    else:
        stats.plot_mean_size("avg","",args.plot)


