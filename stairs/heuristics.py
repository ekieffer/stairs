from deap import gp
from stairs.functions import *
from collections import defaultdict
import operator
import numpy as np
import random

class Heuristic:
    '''
    classdocs
    '''

    def __call__(self ,func):
        raise NotImplementedError


class Dummy(Heuristic):


    def __call__(self,portfolio):


        r = np.random.random()
        if r <= 0.2:
            return 0
        else:
            return 0.01

class Rule1(Heuristic):


    def __call__(self,portfolio):

       return  portfolio.get_distributions(i=0)



class Rule2(Heuristic):


    def __call__(self,portfolio):

        return (portfolio.get_distributions(i=0) + portfolio.get_total_uncalled_capital_for_commitments_older_than(24))


class Rule3(Heuristic):


    def __call__(self,portfolio):

        return (portfolio.get_distributions(i=0) + portfolio.get_total_uncalled_capital_for_commitments_older_than(24)) / (portfolio.get_ID(i=0))

def deZwart(Dt,UCt24,IDt):
    return (Dt + UCt24)*(protectedDiv(1.0,IDt))

def field():
    return defaultdict(list)

#class EvolvableHeuristic(Heuristic):
#
#    def __init__(self, func):
#        self.func = func
#        self.portfolios = defaultdict(field)
#
#    @staticmethod
#    def get_functions_set():
#        pset = gp.PrimitiveSetTyped("MAIN", [float,float,float,float,float,float,float,float,float,float],float)
##        pset = gp.PrimitiveSetTyped("MAIN", [float,float,float,float,float,float,float,float],float)
#        kwargs = {}
#        prefix = "ARG"
#        counter = 0
#        kwargs[prefix + "0"] = "Dt"
#        counter += 1
#        kwargs[prefix + str(counter)] = 'UCt24'
#        counter += 1
#        kwargs[prefix + str(counter)] = 'Casht'
#        counter += 1
#        kwargs[prefix + str(counter)] = 'Ct'
#        counter += 1
#        kwargs[prefix + str(counter)] = 'Navt'
#        counter += 1
#        kwargs[prefix + str(counter)] = 'CCommit'
#        counter += 1
#        kwargs[prefix + str(counter)] = 'IDt'
#        counter += 1
#        kwargs[prefix + str(counter)] = 'error'
#        counter += 1
#        kwargs[prefix + str(counter)] = 'diff_error'
#        counter += 1
#        kwargs[prefix + str(counter)] = 'acc_error'
#        counter += 1
#        pset.addPrimitive(operator.add, [float,float],float)
#        pset.addPrimitive(operator.sub, [float,float],float)
#        pset.addPrimitive(operator.mul, [float,float],float)
#        pset.addPrimitive(protectedDiv, [float,float],float)
#        pset.addPrimitive(inverse, [float],float)
#        pset.addPrimitive(opposite, [float],float)
##        pset.addPrimitive(operator.and_, [bool,bool], bool)
##        pset.addPrimitive(operator.or_, [bool,bool], bool)
##        pset.addPrimitive(operator.ge, [float,float], bool)
##        pset.addPrimitive(operator.le, [float,float], bool)
##        pset.addPrimitive(equal, [float, float], bool)
##        pset.addPrimitive(neg, [bool], bool)
##        pset.addPrimitive(if_then_else, [bool,float,float], float)
##        pset.addTerminal(1,bool)
#        pset.addEphemeralConstant("rand101",lambda: float(random.randint(0, 10)),float)
#        pset.renameArguments(**kwargs)
#        return pset
#
#    def __call__(self,portfolio):
#        results = portfolio.get_all_opti(i=0)
#        nav = results[3]
#        dt = results[5]
#        ct = results[4]
#        cash = results[9]
#        ID = results[10]
#        error = results[11]
#        data = []
##        dt=portfolio.get_distributions(i=0)
#        uct=portfolio.get_total_uncalled_capital_for_commitments_older_than(24)
##        cash=portfolio.get_cash(i=0)
##        ct=portfolio.get_contributions(i=0)
##        nav=portfolio.get_net_asset_value(i=0)
#        commitments=portfolio.get_commitments_since(24)
#
##        ID = portfolio.get_ID(i=0)
##        error = 1.0-ID
#        #################### Assertions
##        assert nav == results[3] # ok
##        assert dt == results[5] # ok
##        assert ct == results[4] # ok
##        assert cash == results[9] # ok
##        assert ID == results[10] # ok
##        assert error == results[11] # ok
#
#
#
#
#
#        ##################### Follow up with derror and serror
#        if len(self.portfolios[id(portfolio)]['error']) == 0:
#            previous_error = 1.0 - portfolio.get_ID(i=1)
#            previous_serror = previous_error
#        else:
#            previous_error = self.portfolios[id(portfolio)]['error'][-1]
#            previous_serror = self.portfolios[id(portfolio)]['serror'][-1]
#        derror = error - previous_error
#        serror = error + previous_serror
#        self.portfolios[id(portfolio)]['ID'].append(ID)
#        self.portfolios[id(portfolio)]['error'].append(error)
#        self.portfolios[id(portfolio)]['derror'].append(derror)
#        self.portfolios[id(portfolio)]['serror'].append(serror)
#        data.append(dt)
#        data.append(uct)
#        data.append(cash)
#        data.append(ct)
#        data.append(nav)
#        data.append(commitments)
#        data.append(ID)
#        data.append(error)
#        data.append(derror)
#        data.append(serror)
#        return self.func(*data)


class EvolvableHeuristic(Heuristic):

    def __init__(self, func):
        self.func = func
        self.portfolios = defaultdict(field)

    @staticmethod
    def get_functions_set():
        pset = gp.PrimitiveSetTyped("MAIN", [float,float,float,float,float,float,float,float,float],float)
        kwargs = {}
        prefix = "ARG"
        counter = 0
        kwargs[prefix + "0"] = "Dt"
        counter += 1
        kwargs[prefix + str(counter)] = 'UCt24'
        counter += 1
        kwargs[prefix + str(counter)] = 'Casht'
        counter += 1
        kwargs[prefix + str(counter)] = 'Ct'
        counter += 1
        kwargs[prefix + str(counter)] = 'Navt'
        counter += 1
        kwargs[prefix + str(counter)] = 'CCommit'
        counter += 1
        kwargs[prefix + str(counter)] = 'IDt'
        counter += 1
        kwargs[prefix + str(counter)] = 'error'
        counter += 1
        kwargs[prefix + str(counter)] = 'deZwart'
        counter += 1
        pset.addPrimitive(operator.add, [float,float],float)
        pset.addPrimitive(operator.sub, [float,float],float)
        pset.addPrimitive(operator.mul, [float,float],float)
        pset.addPrimitive(protectedDiv, [float,float],float)
        pset.addPrimitive(inverse, [float],float)
        pset.addPrimitive(opposite, [float],float)
        pset.addPrimitive(min, [float,float],float)
        pset.addPrimitive(max, [float,float],float)
#        pset.addPrimitive(operator.and_, [bool,bool], bool)
#        pset.addPrimitive(operator.or_, [bool,bool], bool)
#        pset.addPrimitive(operator.ge, [float,float], bool)
#        pset.addPrimitive(operator.le, [float,float], bool)
#        pset.addPrimitive(equal, [float, float], bool)
#        pset.addPrimitive(neg, [bool], bool)
#        pset.addPrimitive(if_then_else, [bool,float,float], float)
#        pset.addTerminal(1,bool)
#        pset.addEphemeralConstant("rand101",lambda: float(random.randint(0, 10)),float)
        pset.renameArguments(**kwargs)
        return pset

    def __call__(self,portfolio):
        data = []

        # Compute data
        dt=portfolio.get_distributions(i=0)
        uct=portfolio.get_total_uncalled_capital_for_commitments_older_than(24)
        cash=portfolio.get_cash(i=0)
        ct=portfolio.get_contributions(i=0)
        nav=portfolio.get_net_asset_value(i=0)
        commitments=portfolio.get_commitments_since(24)
        ID = portfolio.get_ID(i=0)
        error = 1.0-ID

        # Add data
        data.append(dt)
        data.append(uct)
        data.append(cash)
        data.append(ct)
        data.append(nav)
        data.append(commitments)
        data.append(ID)
        data.append(error)
        data.append(deZwart(dt,uct,ID))
        return self.func(*data)










