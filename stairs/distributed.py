import os

class Distributed_Simulation:

      def __init__(self,ipp_client,cashflows,portfolios,cashflow_frequency=4,niters=104,number_of_funds_per_recommitment=1,with_esg=True):
          self.rc = ipp_client
          self.dview = self.rc[:]
          self.dview.block=True
          self.nb_engines = len(self.rc[:])
          self.set_context(cashflows,portfolios,cashflow_frequency,niters,number_of_funds_per_recommitment,with_esg)


      def set_context(self,cashflows,portfolios,cashflow_frequency,niters,number_of_funds_per_recommitment,with_esg):
          current_directory = os.getcwd()
          self.dview.apply_sync(os.chdir,current_directory)
          self.dview.clear(block=True)
#          with self.dview.sync_imports():
#              from stairs.libraries import Library
#              from stairs.simulation import Simulation
#              from stairs.heuristics import EvolvableHeuristic
#              from deap import base, creator, gp
          self.dview.execute("""
              from stairs.libraries import Library
              from stairs.simulation import Simulation
              from stairs.heuristics import EvolvableHeuristic
              from deap import base, creator, gp
          """,block=True)
          self.dview.execute("""
          pset = EvolvableHeuristic.get_functions_set()
          simulation = Simulation("{0}","{1}",cashflow_frequency={2},niters={3},number_of_funds_per_recommitment={4},with_esg={5})
          """.format(cashflows,portfolios,cashflow_frequency,niters,number_of_funds_per_recommitment,with_esg),block=True)



#      def execute(self, heuristics):
#          res=[]
#          indexes = list(range(len(heuristics)))
#          cpt=0
#          while len(indexes) > 0:
#                i = indexes.pop(0)
#                heuristic = heuristics[i]
#                self.dview.push(dict(heuristic=heuristic),block=True,targets=[cpt])
#                cpt += 1
#                if cpt == self.nb_engines:
#                    cpt = 0
#                    result = self.dview.execute("""
#                    func = gp.compile(gp.PrimitiveTree.from_string(heuristic,pset),pset)
#                    score = simulation.execute(EvolvableHeuristic(func))
#                    """,block=True)
#                    res.extend(self.dview['score']))
#          if cpt > 0:
#             result = self.dview.execute("""
#             func = gp.compile(gp.PrimitiveTree.from_string(heuristic,pset),pset)
#             score = simulation.execute(EvolvableHeuristic(func))""",block=True,targets=list(range(cpt)))
#             res.extend(self.dview.pull('score',targets=list(range(cpt))))
#          return res

      def execute(self,heuristics):
          self.dview["heuristics"] = heuristics
          self.dview.scatter("indexes",range(len(heuristics)))
          self.dview.execute("""
          score=[]
          for i in indexes:
              func = gp.compile(gp.PrimitiveTree.from_string(heuristics[i],pset),pset)
              score.append(simulation.execute(EvolvableHeuristic(func)))
          """,block=True)
          return self.dview.gather('score')
         

