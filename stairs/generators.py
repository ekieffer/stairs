import numpy as np
from stairs.portfolios import Portfolio
from stairs.libraries import Library



class PortfolioGenerator():

    def __init__(self,library):
        self.fund_library = library
        self.portfolios = []

    def generate_dataset(self, number_of_portfolios, number_of_funds, overcommitment=0.3, over_years=1):
        """
        Simulated over a year random commitment to number_of_funds

        Parameters
        ----------
        number_of_portfolios: Number of portfolios
        number_of_funds: number of funds for the initialization

        Returns
        -------

        """
        # Still assume quaterly commitments
        # Still need to be updated
        vintages = [i for i in range(over_years * 4) for _ in range(number_of_funds // (over_years * 4))]
        self.portfolios = self.generate_portfolios(number_of_portfolios, number_of_funds,
                                                   overcommitments=overcommitment, vintages=vintages)



    def get_library(self):
        lib=Library()
        for portfolio in self.portfolios:
            lib.append(portfolio)
        return lib


    def train_test_split(self,p):
        assert p > 0 and p <= 1, "Split proportion should in ]0;1]"
        size = len(self.portfolios)
        sample_indexes = np.random.choice(size, int(np.floor(size*p)), False)
        train = Library()
        test = Library()
        for i in range(size):
            if i in sample_indexes:
                train.append(self.portfolios[i])
            else:
                test.append(self.portfolios[i])
        return train,test



    def select_funds_randomly(self,nb):
        return self.fund_library.funds_selection(nb,with_esg=False)


    def clean(self):
        for portfolio in self.portfolios:
            del portfolio
        self.portfolios = []

    def save_portfolios(self,path):
        lib = self.get_library()
        lib.save(path)

    def generate_portfolio(self,number_of_funds,vintages=0,overcommitment=0):
        funds = self.select_funds_randomly(number_of_funds)
        commitments = (np.ones(number_of_funds)+overcommitment)/number_of_funds
        vintage_years=[]
        for k in range(number_of_funds):
            if isinstance(vintages, list):
                # Random overcommitments
                if len(vintages) == 2:
                    a, b = max(vintages[0], vintages[1]), min(vintages[0], vintages[1])
                    vintage = np.random.randint(b,a)
                # All overcommitments are specified
                elif len(vintages) == number_of_funds:
                    vintage = vintages[k]
                # Raise an exception
                else:
                    raise Exception("Incorrect number of item in the vintage list 2 or number_of_funds")
            else:
                # Same vintage year for all
                vintage = vintages
            vintage_years.append(vintage)
        p = Portfolio()
        p.add_funds(funds, commitments, vintage_years)
        p.period = max(vintage_years)
        return p

    def generate_portfolios(self,number_of_portfolios,number_of_funds,vintages,overcommitments=0):
        portfolios = []
        for k in range(number_of_portfolios):
            if isinstance(overcommitments,list):
                # Random overcommitments
                if len(overcommitments) == 2 :
                    a,b=max(overcommitments[0],overcommitments[1]),min(overcommitments[0],overcommitments[1])
                    overcommitment=np.random.random()*(a-b)+b
                # All overcommitments are specified
                elif len(overcommitments) == number_of_portfolios:
                    overcommitment = overcommitments[k]
                # Raise an exception
                else:
                    raise Exception("Error with the overcommitment requests")
            else:
                # Same overcommitment for all portfolios
                overcommitment=overcommitments
            new_portfolio = self.generate_portfolio(number_of_funds,vintages,overcommitment)
            portfolios.append(new_portfolio)
        return portfolios
