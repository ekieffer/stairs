# cython: boundscheck=False
cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython.parallel import prange
import numpy as np
cimport numpy as cnp
from numpy import float64
from numpy cimport float64_t
import pandas



# array = np.asarray(<np.float64_t[:values_length]> values)

cdef struct fund_t:
        float64_t *data
        int rshape
        int lshape
        int vintage
        int end
        float64_t commitment

cdef class Portfolio:


    cdef:
        public float64_t capital
        public  int period 
        fund_t cfunds[500]
        int nb_funds
        public list funds

    def __init__(self):
        self.capital = 1.0
        self.period = 0
        self.nb_funds = 0
        self.funds = list()


    def add_funds(self,list_of_funds,commitments,vintage):
        # Sort fund for vintage
        sorted_indexes = sorted(range(len(vintage)), key = lambda index:vintage[index])
        for index in sorted_indexes:
            self.add_new_fund(commitments[index],list_of_funds[index],vintage[index])


    cdef void add_new_cfund(self,float64_t commitment,float64_t[:,:] fund_data, int vintage, int end):
        self.cfunds[self.nb_funds].data = &fund_data[0,0]
        self.cfunds[self.nb_funds].rshape = fund_data.shape[0]
        self.cfunds[self.nb_funds].lshape = fund_data.shape[1]
        self.cfunds[self.nb_funds].commitment = commitment
        self.cfunds[self.nb_funds].vintage = vintage
        self.cfunds[self.nb_funds].end = end 
        self.nb_funds += 1



    def add_new_fund(self,float64_t commitment,fund, int vintage):
        cdef  int end
        self.funds.append(fund)
        end = vintage + fund.get_param("LIFETIME") -1
        arr = fund.fund_data
        if not arr.flags['C_CONTIGUOUS']:
           arr = np.ascontiguousarray(arr) # Makes a contiguous copy of the numpy array.
        self.add_new_cfund(commitment,arr,vintage,end)



    cdef float64_t [:,::1] cfund_data(self,int i):
         cdef:
             int rshape
             int lshape
             float64_t *data
             int k
             float64_t [:,::1] memview
         data = self.cfunds[i].data
         rshape = self.cfunds[i].rshape            
         lshape = self.cfunds[i].lshape            
         memview = <float64_t[:rshape,:lshape]>data
         return memview

    def get_fund_data(self,int i):
        return np.array(self.cfund_data(i))

    cpdef list get_vintages(self):
        cdef:
            list res = []
            int k
        for k in range(self.nb_funds):
            res.append(self.cfunds[k].vintage)
        return res
         
    cpdef list get_commitments(self):
        cdef:
            list res = []
            int k
        for k in range(self.nb_funds):
            res.append(self.cfunds[k].commitment)
        return res



    cdef float64_t item(self,int k, int i, int j) nogil:
          cdef:
              float64_t value
              int offset
          offset = self.cfunds[k].lshape * i +  j
          return self.cfunds[k].data[offset]

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.nonecheck(False)
    cdef float64_t [:] cget_all_opti(self, int i):
         cdef:
             float64_t *data
             int k
             int j
             float64_t total
             float64_t tmp
            
         results =  np.zeros(12,dtype=float64)
         cdef float64_t [:] results_view = results
         #for k in prange(self.nb_funds,nogil=True,schedule='guided'):
         for k in range(self.nb_funds):
             for j in range(9):
                 tmp=0
                 if i >= self.cfunds[k].vintage and i <= self.cfunds[k].end:
                     tmp = (self.cfunds[k].commitment * self.item(k,j, i - self.cfunds[k].vintage))
                 elif i > self.cfunds[k].end and j >=6 :
                        tmp = (self.cfunds[k].commitment * self.item(k,j, self.cfunds[k].lshape - 1))
                 results_view[j] += tmp
             
         results_view[9] = self.capital - results_view[7] + results_view[8]
         results_view[10] = results_view[3]/(results_view[3] + results_view[9])
         results_view[11] = 1.0 - results_view[10]
         return results_view

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.nonecheck(False)
    def get_all_opti(self, i, relative=True):
        if relative:
            return np.array(self.cget_all_opti(self.period - i))
        else:
            return np.array(self.cget_all_opti(i))

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.nonecheck(False)
    cdef float64_t cget(self, int j,  int i) nogil:
         cdef:
             float64_t *data
             int k
             float64_t total
             float64_t tmp
            
         total=0.0
         #for k in prange(self.nb_funds,nogil=True,schedule='guided'):
         for k in range(self.nb_funds):
             tmp=0
             if i >= self.cfunds[k].vintage and i <= self.cfunds[k].end:
                 tmp = (self.cfunds[k].commitment * self.item(k,j, i - self.cfunds[k].vintage))
             elif i > self.cfunds[k].end and j >=6 :
                    tmp = (self.cfunds[k].commitment * self.item(k,j, self.cfunds[k].lshape - 1))
             total += tmp
         return total


    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.nonecheck(False)
    cdef float64_t [:] _get_all(self, int j):
        cdef  int size = self.get_portfolio_lifetime()
        results =  np.zeros(size,dtype=float64)
        cdef float64_t [:] results_view = results
        cdef  int period
        #for period in prange(size,nogil=True,schedule='guided'):
        for period in range(size):
            results_view[period] = self.cget(j,period)
        return results_view

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    @cython.nonecheck(False)
    def get(self, int j, int i, bint relative):
        if i < 0:
            return np.array(self._get_all(j))
        if relative:
            return self.cget(j, self.period - i)
        else:
            return self.cget(j, i)

    def get_draw_downs(self,i=-1,relative=True):
        """

        Get the capital called at period i or +/- i from current period

        Parameters
        ----------
        i: index
        relative: should be the absolute period or just an offset from the current period

        Returns
        -------
        float
         Capital called at i

        """
        return self.get(0,i,relative)

    def get_management_fees(self,i=-1,relative=True):
        """

        Get the management fees at period i or +/- i from current period

        Parameters
        ----------
        i: index
        relative: should be the absolute period or just an offset from the current period

        Returns
        -------
        float
         Management fees at i

        """
        return self.get(1,i,relative)

    def get_repayments(self,i=-1,relative=True):
        """

        Get the repayements at period i or +/- i from current period


        Parameters
        ----------
        i: index
        relative: should be the absolute period or just an offset from the current period

        Returns
        -------
        float
         Repayments at i

        """
        return self.get(2,i,relative)

    def get_net_asset_value(self,i=-1,relative=True):
        """

        Get the net asset value at period i or +/- i from current period


        Parameters
        ----------
        i: index
        relative: should be the absolute period or just an offset from the current period

        Returns
        -------
        float
         NAV at i

        """
        return self.get(3,i,relative)

    def get_contributions(self,i=-1,relative=True):
        """

        Get the contributions at period i or +/- i from current period


        Parameters
        ----------
        i: index
        relative: should be the absolute period or just an offset from the current period

        Returns
        -------
        float
         Contributions at i

        """
        return self.get(4, i, relative)

    def get_distributions(self,i=-1,relative=True):
        """

        Get the distributions at period i or +/- i from current period


        Parameters
        ----------
        i: index
        relative: should be the absolute period or just an offset from the current period

        Returns
        -------
        float
         Distributions at i

        """
        return self.get(5,i,relative)

    def get_uncalled_capital(self, i=-1, relative=True):
        """

        Get the uncalled_capital at period i or +/- i from current period


        Parameters
        ----------
        i: index
        relative: should be the absolute period or just an offset from the current period

        Returns
        -------
        float
         Uncalled capital at i

        """
        return self.get(6, i, relative)

    def get_cumulated_contributions(self,i=-1,relative=True):
        """

        Get the cumulated contributions at period i or +/- i from current period


        Parameters
        ----------
        i: index
        relative: should be the absolute period or just an offset from the current period

        Returns
        -------
        float
         Uncalled capital at i

        """
        return self.get(7,i,relative)

    def get_cumulated_distributions(self, i=-1, relative=True):
        """

        Get the cumulated distributions at period i or +/- i from current period


        Parameters
        ----------
        i: index
        relative: should be the absolute period or just an offset from the current period

        Returns
        -------
        float
         Uncalled capital at i

        """
        return self.get(8, i, relative)




    def get_cash(self,i=-1,relative=True):
        """

        Get the cash at period i or +/- i from current period


        Parameters
        ----------
        i: index
        relative: should be the absolute period or just an offset from the current period

        Returns
        -------
        float
         Uncalled capital at i

        """
        return self.capital - self.get_cumulated_contributions(i,relative) + self.get_cumulated_distributions(i,relative) 

    def get_ID(self, i=-1, relative=True):
        """

        Get the investment degree at period i or +/- i from current period


        Parameters
        ----------
        i: index
        relative: should be the absolute period or just an offset from the current period

        Returns
        -------
        float
         Uncalled capital at i

        """
        nav = self.get_net_asset_value(i,relative)
        cash = self.get_cash(i,relative)
        return nav/(nav+cash)

    
    def get_error(self,i=-1,relative=True):
        """

        Get the error at period i or +/- i from current period


        Parameters
        ----------
        i: index
        relative: should be the absolute period or just an offset from the current period

        Returns
        -------
        float
         error at i

        """
        return 1.0-self.get_ID(i,relative)


    def get_derror(self,i=-1,relative=True):
        """

        Get the differential error at period i or +/- i from current period


        Parameters
        ----------
        i: index
        relative: should be the absolute period or just an offset from the current period

        Returns
        -------
        float
         error at i

        """
        if i < 0:
            return np.diff(self.get_error(i=-1))
        if relative:
            return self.get_error(i,relative=True) - self.get_error(i+1,relative=True)
        else:
            return self.get_error(i,relative=False) - self.get_error(i-1,relative=False)

    def get_serror(self,i=-1,relative=True):
        """

        Get the accumulated error at period i or +/- i from current period


        Parameters
        ----------
        i: index
        relative: should be the absolute period or just an offset from the current period

        Returns
        -------
        float
         error at i

        """
        if i < 0:
            return np.cumsum(self.get_error(i=-1))
        acc = 0
        if relative:
            end = self.period - i
        else:
            end = i
        for k in range(end+1):
            acc += self.get_error(i=k,relative=False)
        return acc




    def __len__(self):
        return len(self.funds)


    def __eq__(self,other):
        """

        Compare equality of two portfolios.
        This will clearly take time if there are a lot of portfolios to compare.
        In any case, it should only be seldom used, i.e. just to generate the original dataset

        Parameters
        ----------
        other

        Returns
        -------

        """
        return self.funds == other.funds


    def step(self):
        self.period += 1
        return self.period


    cdef  int cget_end_portfolio(self):
       cdef  int k
       cdef  int max_value = 0
       for k in range(self.nb_funds):
           if self.cfunds[k].end > max_value:
               max_value = self.cfunds[k].end
       return max_value

    def get_end_portfolio(self):
        return self.cget_end_portfolio()

    def is_ended(self):
        return self.period > self.get_end_portfolio()


    def get_portfolio_lifetime(self):
        return self.get_end_portfolio()+1


    cdef  int cget_end_fund(self, int i):
         return self.cfunds[i].end

    def get_end_fund(self, int i):
        return self.cget_end_fund(i)


    cdef  int cget_fund_age(self, int i):
         return self.period - self.cfunds[i].vintage + 1

    def get_fund_age(self,  int i):
        return self.cget_fund_age(i)

    def is_fund_ended(self, int i):
        return self.period > self.cget_end_fund(i)


    cdef float64_t cget_uncalled_capital_for_commitments_older_than(self,  int period):
        cdef:
            float64_t total
            int k
            int vintage
            int when
            double *data
            float64_t [:,::1] memview
        when = self.period - period
        if when <= 0:
            return 0
        total = 0.0
        for k in range(self.nb_funds):
            data = self.cfunds[k].data
            rshape = self.cfunds[k].rshape            
            lshape = self.cfunds[k].lshape            
            memview = <float64_t[:rshape,:lshape]>data
            vintage = self.cfunds[k].vintage
            if vintage == when:
                total += (self.cfunds[k].commitment * memview[6,self.period-vintage])
        return total

    cdef float64_t cget_commitments_since(self,  int period):
        cdef:
            float64_t total
            int k
            int vintage
            int when
            double *data
            float64_t [:,::1] memview
        when = self.period - period
        if when <= 0:
            return 0
        total = 0.0
        for k in range(self.nb_funds):
            data = self.cfunds[k].data
            rshape = self.cfunds[k].rshape            
            lshape = self.cfunds[k].lshape            
            memview = <float64_t[:rshape,:lshape]>data
            vintage = self.cfunds[k].vintage
            if vintage >= when:
                total += self.cfunds[k].commitment 
        return total


    def get_total_uncalled_capital_for_commitments_older_than(self, period):
        """
        Uncalled capital for commiments made period ago
        """
        return self.cget_uncalled_capital_for_commitments_older_than(period)

    def get_commitments_since(self,period):
        return self.cget_commitments_since(period)


    cdef list make_commitments_list(self):
         cdef  int k
         commitments = []
         for k in range(self.nb_funds):
             commitments.append(self.cfunds[k].commitment)
         return commitments


    cdef list make_vintages_list(self):
         cdef  int k
         vintages = []
         for k in range(self.nb_funds):
             vintages.append(self.cfunds[k].vintage)
         return vintages



    def __getstate__(self):
        """

        INFO: Still need to add fund info


        Returns
        -------

        """
        state = { "capital":self.capital,"commitments":self.make_commitments_list(), "vintages":self.make_vintages_list(),"funds":[self.funds[i] for i in range(len(self.funds))]}
        return state

    def __setstate__(self, state):
        cdef int k
        funds = state['funds']
        commitments = state['commitments']
        vintages = state['vintages']
        #self.capital = state['capital']
        self.capital = 1.0
        self.period = vintages[-1]
        self.nb_funds = 0
        self.funds = []
        for k in range(len(funds)):
            self.add_new_fund(commitments[k],funds[k],vintages[k])
            
    def clone(self):
        portfolio = Portfolio()
        portfolio.__setstate__(self.__getstate__())
        return portfolio


















