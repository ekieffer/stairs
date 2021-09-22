'''
Created on Jul 13, 2017

@author: manu
'''


from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from datetime import datetime
from stairs.heuristics import EvolvableHeuristic
from stairs.distributed import Distributed_Simulation
from torch.utils.tensorboard import SummaryWriter
import operator
import random
import numpy as np



class GPLearner(object):
    '''
    classdocs
    '''

    def __init__(self,simulator,pset,npop,cxpb,mutpb,ngen,tf_logsdir):
        '''
        Constructor
        '''
        self.simulator=simulator
        self.npop=npop
        self.ngen=ngen
        self.cxpb=cxpb
        self.mutpb=mutpb
        self.pset = pset
        self.gp_initialization()
        self.generated_functions=[]
        self.writer = None
        if tf_logsdir is not None:
            self.writer = SummaryWriter(tf_logsdir)



  
    def evaluate(self,individual):
        '''
        GP evaluation function
        '''
        func = self.toolbox.compile(expr=individual)
        score = self.simulator.execute(EvolvableHeuristic(func))
        return score

    def gp_initialization(self):
        creator.create("FitnessMinGP", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMinGP)
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=5)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("evaluate", self.evaluate)
        #self.toolbox.register("select", tools.selTournament, tournsize=2)
        self.toolbox.register("select", tools.selTournament,tournsize=2)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=5)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
    
    def gp_evolution(self,population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
        """This algorithm reproduce the simplest evolutionary algorithm as
        presented in chapter 7 of [Back2000]_.
    
        :param population: A list of individuals.
        :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                        operators.
        :param cxpb: The probability of mating two individuals.
        :param mutpb: The probability of mutating an individual.
        :param ngen: The number of generation.
        :param stats: A :class:`~deap.tools.Statistics` object that is updated
                      inplace, optional.
        :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                           contain the best individuals, optional.
        :param verbose: Whether or not to log the statistics.
        :returns: The final population
        :returns: A class:`~deap.tools.Logbook` with the statistics of the
                  evolution
    
        The algorithm takes in a population and evolves it in place using the
        :meth:`varAnd` method. It returns the optimized population and a
        :class:`~deap.tools.Logbook` with the statistics of the evolution. The
        logbook will contain the generation number, the number of evalutions for
        each generation and the statistics if a :class:`~deap.tools.Statistics` is
        given as argument. The *cxpb* and *mutpb* arguments are passed to the
        :func:`varAnd` function. The pseudocode goes as follow ::
    
            evaluate(population)
            for g in range(ngen):
                population = select(population, len(population))
                offspring = varAnd(population, toolbox, cxpb, mutpb)
                evaluate(offspring)
                population = offspring
    
        As stated in the pseudocode above, the algorithm goes as follow. First, it
        evaluates the individuals with an invalid fitness. Second, it enters the
        generational loop where the selection procedure is applied to entirely
        replace the parental population. The 1:1 replacement ratio of this
        algorithm **requires** the selection procedure to be stochastic and to
        select multiple times the same individual, for example,
        :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
        Third, it applies the :func:`varAnd` function to produce the next
        generation population. Fourth, it evaluates the new individuals and
        compute the statistics on this population. Finally, when *ngen*
        generations are done, the algorithm returns a tuple with the final
        population and a :class:`~deap.tools.Logbook` of the evolution.
    
        .. note::
    
            Using a non-stochastic selection method will result in no selection as
            the operator selects *n* individuals from a pool of *n*.
    
        This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
        :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
        registered in the toolbox.
    
        .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
           Basic Algorithms and Operators", 2000.
        """
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        if isinstance(self.simulator,Distributed_Simulation):
            fitnesses = self.simulator.execute([str(ind) for ind in invalid_ind])
        else:
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            print("Ind: {0}, score: {1}".format(str(ind),str(fit)),flush=True)
            ind.fitness.values = fit
    
        if halloffame is not None:
            halloffame.update(population)
    
        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if self.writer is not None:
            self.writer.add_scalars('Convergence',{'min':record['fitness']['min'],'avg':record['fitness']['avg'],'max':record['fitness']['max']},0)
            self.writer.flush()
        if verbose:
            print(logbook.stream)
        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))
    
            # Vary the pool of individuals
            offspring = algorithms.varOr(offspring, toolbox, len(population) ,cxpb, mutpb)
    
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if isinstance(self.simulator,Distributed_Simulation):
                fitnesses = self.simulator.execute([str(ind) for ind in invalid_ind])
            else:
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)
    
            # Replace the current population by the offspring
            population[:] = toolbox.select(population+offspring,len(population))
    
            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if self.writer is not None:
                self.writer.add_scalars('Convergence',{'min':record['fitness']['min'],'avg':record['fitness']['avg'],'max':record['fitness']['max']},gen)
                self.writer.flush()
            if verbose:
                print(logbook.stream)
#             diversity = logbook.chapters['fitness'].select("std")
#             if diversity[-1] < 1e-4:
#                 break
        if self.writer is not None:
            self.writer.close()
        return population, logbook
    
    
      
    # record=(path,run)    
    def optimize(self,nHof,warm_start):
        random.seed(datetime.now())
        pop = self.toolbox.population(n=self.npop-len(warm_start))
        hof = tools.HallOfFame(1)
        #hof = tools.ParetoFront()
        for item in warm_start:
            ind=creator.Individual(gp.PrimitiveTree.from_string(item,self.pset))
            pop.append(ind)

        
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean, axis=0)
        mstats.register("std", np.std,  axis=0)
        mstats.register("min", np.min,  axis=0)
        mstats.register("max", np.max,  axis=0)
    
        pop, log = self.gp_evolution(pop, self.toolbox, self.cxpb, self.mutpb,self.ngen, stats=mstats,
                                       halloffame=hof, verbose=True)

        # print log 
        return pop,log,hof
    

    def learn(self,nhof=1,warm_start=[]):
        self.generated_functions = []
        pop,log,hof = self.optimize(nhof,warm_start)
        for i in range(len(hof)):
            self.generated_functions.append(self.toolbox.compile(hof[i]))
        return pop,log,hof


    # def plot_tree(self,ind):
    #
    #     import matplotlib.pyplot as plt
    #     import networkx as nx
    #     nodes, edges, labels = gp.graph(ind)
    #     g = nx.Graph()
    #     g.add_nodes_from(nodes)
    #     g.add_edges_from(edges)
    #     pos = nx.nx_pydot.graphviz_layout(g, prog="dot")
    #
    #     nx.draw_networkx_nodes(g, pos)
    #     nx.draw_networkx_edges(g, pos)
    #     nx.draw_networkx_labels(g, pos, labels)
    #     plt.show()
    #

    

        
    
