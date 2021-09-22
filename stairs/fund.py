from stairs.utils import barplot,lineplot
import numpy as np
import matplotlib.pyplot as plt
import pandas
import zlib



class Fund:
    """
    Model a private equity fund (read-only)
    """

    def __init__(self):
        self.fund_data = None
        # New member added from Cashflows provided by T.Meyer in Dec. 2020
        self.fund_params = None
        self.period = 0


    def load_from_generator(self, **kwargs):

        draw_downs = kwargs.get("draw_downs",None)
        mgmt_fees = kwargs.get("mgmt_fees",None)
        repayments = kwargs.get("repayments",None)
        fixed_returns = kwargs.get("fixed_returns",None)
        values = kwargs.get("values",None)
        parameters = kwargs.get("parameters",None)
        uncalled_capital = kwargs.get("uncalled_capital",None)

        # Checks
        assert draw_downs is not None
        assert mgmt_fees is not None
        assert repayments is not None
        #assert fixed_returns is not None
        assert values is not None
        assert uncalled_capital is not None

        # Checks data size
        sizes= [len(draw_downs), len(mgmt_fees),len(repayments),
                len(values)]
        assert min(sizes) == max(sizes)


        """

        Parameters
        ----------
        draw_downs: capital called
        mgmt_fees: management fees
        repayments: distributions
        values: NAV

        """

        contributions = draw_downs + mgmt_fees
        if fixed_returns is not None:
            # Check that if they are repayments, they have the same size of the repayment
            # Just a practical aspect
            assert len(fixed_returns) == len(repayments)
            distributions = repayments + fixed_returns

        uncalled_capital = 1.0 - np.cumsum(contributions)

        self.fund_data = np.vstack((draw_downs,  # 0
                                    mgmt_fees,  # 1
                                    repayments,  # 2
                                    values,  # 3
                                    contributions, # 4
                                    distributions, # 5
                                    uncalled_capital, #6
                                    np.cumsum(contributions), #7
                                    np.cumsum(distributions),  #8
                                    ))


        self.fund_params = parameters


    def get_param(self,key):
        """

        Parameters
        ----------
        key: parameter name

        Returns
        -------
         The corresponding parameter value
        """
        return self.fund_params.get(key,None)

    def set_param(self,key,value):
        """

        Parameters
        ----------
        key: parameter name
        value: parameter value

        Returns
        -------
         None
        """
        self.fund_params[key] = value

    def get_cashflow_periodicity(self):
        request = self.get_param('PERIODICITY')
        if request is not None:
            if request == 'QUARTERLY':
                return 4
            elif request == 'MONTHLY':
                return 12
            else:
                # Yearly
                return 1
        else:
            # We need to guess
            # Fund nearly 12 years
            length = self.fund_data.shape[-1]
            periods = np.array([4,12,1])
            return np.argmin(np.abs((length // periods)-12))





    def load_from_numpy(self,numpy_matrix,parameters=None):
        """

        Parameters
        ----------
        numpy_matrix: numpy matrix representing the fund data

        """
        self.fund_data=numpy_matrix
        self.fund_params=parameters
        self.reset()

    def is_valid(self):
        """

        Verity that the fund is valid, i.e., the fund data is not empty

        Returns
        -------
        boolean
         True if fund data is existing else None

        """
        return self.fund_data is not None

    def reset(self):
        """

        Reset the observation period

        Returns
        -------

        """
        self.period=0

    def is_terminated(self):
        """

        Verify if the the period reach the end of the fund lifetime

        Returns
        -------
        boolean
         True if period points to last column of the fund data matrix

        """
        assert (self.fund_data is not None)
        return self.period >= self.fund_data.shape[-1]

    def get_lifetime(self):
        """

        Duartion of a fund

        Returns
        -------
        int
         The lifetime of the fund (can be quarter,months,years)

        """
        return self.fund_data.shape[-1]

    def get_period(self):
        """

        Current period

        Returns
        -------
        float
         Provide the current period (can be quarter,month,year)

        """
        assert (self.fund_data is not None)
        return self.period

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
        return self.get(self.fund_data[0,:],i,relative)

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
        return self.get(self.fund_data[1,:],i,relative)

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
        return self.get(self.fund_data[3,:],i,relative)

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
        return self.get(self.fund_data[3,:],i,relative)

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
        return self.get(self.fund_data[4, :], i, relative)

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
         Contributions at i

        """
        return self.get(self.fund_data[7, :], i, relative)

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
        return self.get(self.fund_data[5,:],i,relative)


    def get_cumulated_distributions(self,i=-1,relative=True):
        """

        Get the cumulated distributions at period i or +/- i from current period


        Parameters
        ----------
        i: index
        relative: should be the absolute period or just an offset from the current period

        Returns
        -------
        float
         Distributions at i

        """
        return self.get(self.fund_data[8,:],i,relative)

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
        return self.get(self.fund_data[6, :], i, relative)

    def get(self,data,i,relative):
        """
        Retrieve fund data
        i: index or offset
        i < 0 --> get all data (no indexing)
        i=1 --> relative = True  offset in the past
        i=1 --> relative = False  absolute index


        Parameters
        ----------
        i: index or offset
        relative: should be the absolute period or just an offset from the current period

        Returns
        -------
        float
         data at i

        """
        if i < 0 :
            return data
        if relative:
            assert(self.period - i >= 0 and i >= 0 and self.fund_data is not None), "Error: offset i is incorrect"
            return data[self.period - i]
        else:
            duration=self.get_lifetime()
            assert (i >= 0 and i <= duration  and self.fund_data is not None), "Error: index i incorrect"
            return data[i]

    def clone(self):
        """

        Clone a fund

        Returns
        -------
        Fund
         A new copy of the fund

        """
        new_fund = Fund()
        array_matrix = np.array(self.fund_data, copy=True)
        param_copy = self.fund_params.copy() if self.fund_params is not None else None
        new_fund.load_from_numpy(array_matrix,parameters=param_copy)
        return new_fund

    def scale(self, commitment_value):
        """

        Fund data have been generated to represent a commitment of 1.


        Parameters
        ----------
        commitment_value: a value between 0 and 1

        """
        #assert(commitment_value > 0 and commitment_value <= 1)
        assert (commitment_value > 0)
        self.reset()
        self.fund_data = commitment_value * self.fund_data

    def step(self):
        """

        Step in the fund lifetime by increasing the current observation period

        """
        assert(self.fund_data is not None)
        if self.period < self.get_lifetime():
            self.period += 1

    def __hash__(self):
        """

        Generate a hash for the fund

        Returns
        -------
        str
         A hash value

        """
        assert(self.fund_data is not None)
        return zlib.crc32(bytes(self.fund_data))

    def __eq__(self, other):
        """

        Test fund equality to another one


        Parameters
        ----------
        other: Another fund

        Returns
        -------
        boolean
         True if funds are equals else False

        """
        return self.__hash__() == other.__hash__()

    def __getstate__(self):
        """

        Create a state for teh fund.


        """
        assert(self.fund_data is not None)
        state = {"data":self.fund_data, "params":self.fund_params}
        return state

    def __setstate__(self, state):
        """

        Set the fund state

        Parameters
        ----------
        state

        """
        self.fund_data = state["data"]
        self.fund_params= state["params"]


    def plot_draw_downs(self):
        """

        Plot draw downs

        """
        barplot(-1.0 * self.fund_data[0, :], "Draw downs")

    def plot_management_fees(self):
        """

        Plot management fees

        """
        barplot(self.fund_data[1, :], "Management fees")

    def plot_repayments(self):
        """

        Plot repayments

        """
        barplot(self.fund_data[2, :], "Distributions")

    def plot_net_asset_value(self,**kwargs):
        """

        Plot net asset values

        """
        lineplot(self.fund_data[3, :], "Net Asset Value",**kwargs)

    def plot_contributions(self):
        """

        Plot contributions


        """
        lineplot(self.fund_data[4, :], "Contributions")

    def plot_distributions(self):
        """

        Plot distributions


        """
        lineplot(self.fund_data[5, :], "Distributions")

    def plot_cumulative_contribution(self):
        """

        Plot cumulatives contributions


        """
        barplot(np.cumsum(-1.0 * self.fund_data[0, :] - self.fund_data[1, :]), "Cumulative contributions")

    def plot_cumulative_distribution(self):
        """

        Plot cumulatives distributions


        """
        barplot(self.fund_data[5, :], "Cumulative distributions")

    def plot_uncalled_capital(self):
        """

        Plot uncalled capital

        """
        xaxis = np.arange(self.fund_data.shape[1])
        p1 = plt.fill_between(xaxis, np.ones_like(xaxis) ,color="black")
        p2 = plt.fill_between(xaxis, np.cumsum(self.fund_data[4,:]),color="white")
        plt.legend((p1, p2), ("Remaining Cash", "Cumulated contributions"), loc="lower right")
        plt.title("Cash evolution")
        plt.show()

    def plot_fraction_of_total_commitment(self,title,save=None):
        """

        Plot the fraction of the total commitment


        """
        plt.clf()
        p1, = plt.plot(np.cumsum(self.fund_data[4,:]))
        p2, = plt.plot(np.cumsum(self.fund_data[5,:]))
        p3, = plt.plot(self.fund_data[3, :])
        plt.legend((p1, p2, p3), ("Cumulated contributions", "Cumulated distributions", "Net Value"), loc="best")
        plt.title(title)
        if save is not None:
            plt.savefig(save,dpi=600,format='png')
            return
        plt.show()

    def plot_total_fund_capital(self):
        """

        Plot total fund capital

        """
        xaxis = np.arange(self.fund_data.shape[1])
        nav = self.fund_data[3,:]
        uc = self.fund_data[6,:]
        distrib = np.cumsum(self.fund_data[5,:])
        total = nav + uc + distrib
        p3 = plt.fill_between(xaxis, np.ones_like(xaxis) ,color="black")
        p2 = plt.fill_between(xaxis, (distrib+nav)/total, color="grey")
        p1 = plt.fill_between(xaxis, nav/total, color="white")
        plt.xticks(xaxis, ["Vintage {0}".format(k) for k in range(self.fund_data.shape[1])])
        plt.legend((p1,p2,p3),("Net Asset Value","Cash from distributions","Cash"),loc="best")
        plt.title("Total fund capital")
        plt.show()

    def plot_jcurve(self):
        """

        Plot the Jcurve

        """
        net_cashflow= self.fund_data[5, :] - self.fund_data[4, :]
        barplot(np.cumsum(net_cashflow), "Jcurve IRR={0}".format(np.irr(net_cashflow)))

    def __str__(self):
        """

        String representaion of the cashflow

        Returns
        -------
        str
          the Jcurve as a string

        """
        net_cashflow= self.fund_data[5, :] - self.fund_data[4, :]
        return np.array2string(np.cumsum(net_cashflow))


    def exportToExcel(self,path):
        data=["draw_downs",  # 0
              "mgmt_fees",  # 1
              "repayments",  # 2
              "values",  # 3
              "contributions",  # 4
              "distributions",  # 5
              "uncalled_capital"]  # 6
        with pandas.ExcelWriter(path) as writer:
             df=pandas.DataFrame(self.fund_data)
             df.to_excel(writer,sheet_name="data",float_format="%.4f")










