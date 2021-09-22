from stairs.libraries import Library
from stairs.generators import PortfolioGenerator
import argparse

def main(**kwargs):
    lib = Library()
    lib.load_library(kwargs["path_lib_cashflows"])

    gen = PortfolioGenerator(lib)
    gen.generate_dataset(kwargs["number_portfolios"], kwargs["number_initial_funds"], overcommitment=kwargs["overcommitment"], over_years=kwargs["period"])
    gen.save_portfolios(kwargs["path_save_portfolios"])



if __name__ == "__main__":
    kwargs = dict()
    kwargs["path_lib_cashflows"] = None
    kwargs["path_save_portfolios"] = None
    kwargs["number_portfolios"] = 100 
    kwargs["number_initial_funds"] = 16 
    kwargs["overcommitment"] = 0
    kwargs["period"] = 1
	
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--portfolios', type=int, default=kwargs["number_portfolios"]  ,help="Number of initial portfolios")
    parser.add_argument('-f', '--funds', type=int, default=kwargs["number_initial_funds"], help="Number of initial funds")
    parser.add_argument('-o', '--overcommitment', type=float, default=kwargs["overcommitment"], help="Overcommitment")
    parser.add_argument('-t', '--time', type=float, default=kwargs["period"], help="Initial portfolios over time")
    parser.add_argument('path_lib_cashflows')
    parser.add_argument('path_save_portfolios')
    args = parser.parse_args()
    kwargs["path_lib_cashflows"] = args.path_lib_cashflows
    kwargs["path_save_portfolios"] = args.path_save_portfolios
    kwargs["number_portfolios"] = args.portfolios
    kwargs["number_initial_funds"] = args.funds
    kwargs["overcommitment"] = args.overcommitment
    kwargs["period"] = args.time
    main(**kwargs)

	

