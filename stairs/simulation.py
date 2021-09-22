import os
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"]="0"
from stairs.fund import *
from stairs.libraries import Library
from stairs.heuristics import *
from stairs.heuristicLearners import *
from stairs.exceptions import CashShortage,CashFlows_Freq
from stairs.utils import limits
import matplotlib.pyplot as plt
from numpy.testing import assert_almost_equal

class Simulation:

    def __init__(self,cashflows,portfolios,
                              cashflow_frequency=4,
                              niters=104,
                              number_of_funds_per_recommitment=1,
                              with_esg=True):

        # Simulation parameters
        self.niters = niters
        self.number_of_funds_per_recommitment = number_of_funds_per_recommitment
        self.with_esg = with_esg
        #####################


        self.__i__ = -1
        self.funds_library = Library()
        self.funds_library.load_library(cashflows)
        self.lib_portfolios = Library()
        self.lib_portfolios.load_library(portfolios)
        if not self.lib_portfolios.arePortfoliosSelected():
           self.lib_portfolios.switch()
        self.portfolios = []
        self.invalid_portfolios = {}
        # By default, we consider quarterly cashflows
        self.cashflow_frequency=cashflow_frequency
        self.start_commitment = np.max([p.period for p in self.lib_portfolios])
        self.end_commitment = self.start_commitment + self.niters
        self.reset()



    def __getitem__(self, index):
        if index  < len(self.portfolios):
            return self.portfolios[index]
        else:
            raise IndexError("list index out of range")

    def __iter__(self):
        return self

    def __next__(self):
        self.__i__ += 1
        if self.__i__ < len(self.portfolios):
            return self.portfolios[self.__i__]
        else:
            self.__i__=-1
            raise StopIteration


    def check_cashflows_periodicity(self):
        bad_freq = {}
        for i,fund in enumerate(self.funds_library.funds_set):
            freq = fund.get_cashflow_periodicity()
            if freq != self.cashflow_frequency:
               bad_freq[i] = freq
        if len(bad_freq) > 0:
            raise CashFlows_Freq("Cashflows have not the right frequency")

    def select_funds(self,number_of_funds):
        return self.funds_library.funds_selection(number_of_funds,self.with_esg)

    def reset(self):
        for portfolio in self.portfolios:
            del portfolio
        if not self.lib_portfolios.arePortfoliosSelected():
                self.lib_portfolios.switch()
        self.portfolios = self.lib_portfolios.get_list_copy_of_set()
        self.invalid_portfolios = {}

    def execute(self,heuristic):
        self.reset()
        try:
            self.check_cashflows_periodicity()
        except CashFlows_Freq as e:
            print(e)
            # Procedure to update the frequencies
            # remove from lib or adapty cashflow
        # Makes no sense init a single time
        ids = np.empty((0,self.niters))
        invalids = 0
        for index in range(len(self.portfolios)):
            try:
                self.simulate_single_portfolio(index, heuristic, self.number_of_funds_per_recommitment)
            except CashShortage as cs:
                invalids += 1
            ID = self.portfolios[index].get_ID(i=-1)
            shortage_position = np.where(ID > 1.0)[0]
            if len(shortage_position) > 0:
               # Portfolio with cash shortage
               min_pos = shortage_position.min()
               ID = ID[self.start_commitment:min(min_pos,self.end_commitment)]
               ids = np.vstack((ids,np.pad(ID,(0,self.niters - len(ID)),'constant',constant_values=0)))
            elif len(ID) < self.end_commitment:
                ID = ID[self.start_commitment:]
                ids = np.vstack((ids,np.pad(ID,(0,self.niters - len(ID)),'constant',constant_values=0)))
            else:
                ID = ID[self.start_commitment:self.end_commitment]
                ids = np.vstack((ids,ID))
        ### Invalids
        n = len(self.portfolios)
        p = invalids / n
        q = 1.0-p
        sig_freq = np.sqrt(p*q/n)
        ### ID
        UB = np.mean(ids,axis=0) + 1.96 *(np.std(ids,axis=0)/np.sqrt(n))
        return np.sum(np.abs(1.0-UB)), #np.sum(np.std(ids,axis=0)/np.sqrt(n)) #p + 1.96 * sig_freq



    def preprocessing_single_portfolio(self,index,**kwargs):
        portfolio = self.portfolios[index]
        cash = portfolio.get_cash(i=0)
        if cash <= 0:
           raise CashShortage(cash)


    def postprocessing_single_portfolio(self,index,**kwargs):
        pass

    def simulate_single_portfolio(self,index,heuristic,number_of_funds):
        iter_commitment=0
        while not self.portfolios[index].is_ended():
            self.preprocessing_single_portfolio(index)
            # Ok we increase the period
            self.portfolios[index].step()
            if iter_commitment < self.niters:
                self.recommit_single_portfolio(index,heuristic,number_of_funds)
                iter_commitment += 1
            self.postprocessing_single_portfolio(index)


    def recommit_single_portfolio(self, index, heuristic, number_of_funds):
        """
        This is the main method to perform the recommitments.

        Parameters
        ----------
        index: Portfolio index in the simulation
        heuristic: heuristic to use for the recommitment
        number_of_funds: number_of_funds to recommit into

        Returns
        -------
        The recommitment value
        """


        portfolio = self.portfolios[index]
        total_recommitment = limits(heuristic(portfolio),1e-4,1e+4)
        # Commit or not we continue
        if total_recommitment > 1e-4:

            # Generate vintages
            vintages = [portfolio.period ] * number_of_funds

            recommitment = total_recommitment / number_of_funds
            funds = self.select_funds(number_of_funds)
            portfolio.add_funds(funds, [recommitment]*number_of_funds, vintages)


    def dummy_step(self):
        for index in range(len(self.portfolios)):
            self.portfolios[index].step()



    # All functions below should be placed in other class for posproceesing

    def plot_ID(self,strategy=np.mean):
        min_age = min([self.portfolios[i].get_end_portfolio() for i in range(len(self.portfolios))])
        ids = np.empty((0,min_age))
        for i in range(len(self.portfolios)):
                ids = np.vstack((ids,self.portfolios[i].get_ID(i=-1)[:min_age]))
        lineplot(strategy(ids,axis=0),"Mean ID",color="black",linewidth=4)

    def plot_IDs(self):
        for i in range(len(self.portfolios)):
                plt.plot(self.portfolios[i].get_ID(i=-1))
        plt.show()



                    







          
          





if __name__ == "__main__":
    lib = Library()
    #lib.load_library("/home/manu/Documents/Work/HPC/Repositories/git/lab.uni.lu/ulhpc-ekieffer/stairs-code/data/lib.h5")
    lib.load_library("../data/deZwart/lib_cashflows_deZwart.h5")
    #lib.load_library("/home/manu/Documents/Work/HPC/Repositories/git/lab.uni.lu/ulhpc-ekieffer/stairs-code/data/lib_cashflows.h5")
    #lib.load_library("/home/manu/Documents/Work/HPC/Repositories/git/lab.uni.lu/ulhpc-ekieffer/stairs-code/data/deZwart/bench_deZwart2.h5")
    #lib[0].exportToExcel("/home/manu/Documents/Work/HPC/Repositories/git/lab.uni.lu/ulhpc-ekieffer/stairs-code/results/fund.xlsx")
    #lib.load_library("/home/manu/Documents/Work/HPC/Repositories/git/lab.uni.lu/ulhpc-ekieffer/stairs-code/data/lib_cashflows_mean_4.633_var_0.056.h5")
    #lib.load_library("/home/manu/Documents/Work/HPC/Repositories/git/lab.uni.lu/ulhpc-ekieffer/stairs-code/data/lib_cashflows_mean_0.959_var_0.084.h5")
    lib_training = Library()
    lib_training.load_library("/home/manu/Documents/Work/HPC/Repositories/git/lab.uni.lu/ulhpc-ekieffer/stairs-code/data/lib_no_commitments.h5")
    if not lib_training.arePortfoliosSelected():
        lib_training.switch()

    p = lib_training[0]
    print(len(p.get_error()))
    print(len(p.get_serror()))
    import sys
    sys.exit(1)


    s = Simulation("../data/deZwart/lib_cashflows_deZwart.h5",
                   "/home/manu/Documents/Work/HPC/Repositories/git/lab.uni.lu/ulhpc-ekieffer/stairs-code/data/lib_no_commitments.h5",
                   niters=104,number_of_funds_per_recommitment=1)

    

    pset = EvolvableHeuristic.get_functions_set()
    expr="mul(mul(add(Dt,UCt24),inverse(IDt)), Dt)"
    func = gp.compile(gp.PrimitiveTree.from_string(expr,pset),pset)
    print(s.execute(EvolvableHeuristic(func)))

    #p=Profiler(s)
    #p.execute(Rule1())
    s.plot_IDs()
    s.plot_ID(strategy=np.mean)


    

   # from deap import gp
   # pset = EvolvableHeuristic.get_functions_set()
   # func = gp.compile(gp.PrimitiveTree.from_string("add(Dj, UCt24)",pset),pset)
   # start=time.time()
   # #recommitments = s.execute(EvolvableHeuristic(func))
   # recommitments = s.execute(Rule1())
   # print("Time: {0}".format(time.time() - start))


























