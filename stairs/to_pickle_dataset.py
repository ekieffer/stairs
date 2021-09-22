from stairs.libraries import Library
from stairs.generators import PortfolioGenerator
import argparse
import pickle
import logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',datefmt='%Y-%m-%d %H:%M:%S')

def main(paths):
    logging.info("Converting h5 to pickle") 
    for p in paths:
        logging.info("----> Loading: {0}".format(p)) 
        lib = Library()
        lib.load_library(p)
        with open(p.replace("h5","lib"),"wb") as fd:
            pickle.dump(lib,fd)
        logging.info("----> Done: {0}".format(p.replace("h5","lib"))) 




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(nargs="+", dest="paths", help="One or more h5 library")
    args = parser.parse_args()
    main(args.paths)

	

