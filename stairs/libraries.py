import ast
import random
import sys
import pickle
import numpy as np
import h5py
from sklearn.mixture import GaussianMixture
import  matplotlib.pyplot as plt
from stairs.utils import  bisect_left,bisect_right
from stairs.fund import Fund
from stairs.portfolios import Portfolio
from stairs.utils import listSet, to_proba_scores
from scipy.stats import norm, describe

class Library:
    """
    A collection of funds
    """

    def __init__(self):
        ## Dictionnary because we assume that they are all different
        self.funds_set=listSet()
        self.portfolios_set=[]
        self.set = self.funds_set
        self.__i__=-1
        self.annotations={}
        # ESG cumulative weights
        self.esg_cum_weights = None 


    def set_annotation(self,key,value):
        self.annotations[key]=value

    def get_annotation(self,key):
        return self.annotations[key]


    def switch(self):
        """

        You need to swith to the right set if you want to iterate over it


        Returns
        -------

        """
        if self.set == self.funds_set:
            self.set = self.portfolios_set
        else:
            self.set = self.funds_set

    def areFundsSelected(self):
        return self.set == self.funds_set

    def arePortfoliosSelected(self):
        return self.set == self.portfolios_set

    def append(self,object):
        """

        Add a fund to the collection

        Parameters
        ----------
        fund: a fund

        Returns
        -------

        None

        """
        if isinstance(object,Portfolio):
            self.portfolios_set.append(object)
            for fund in object.funds:
                self.funds_set.append(fund)
        else:
            self.funds_set.append(object)


    def rev(self):
        """

        Reverse ordering of funds

        Returns
        -------

        """
        self.set.reverse()


    def shuffle(self):
        """

        Shuffle the list of funds

        Returns
        -------

        """
        for i in range(10):
            random.shuffle(self.set)


    def save(self,file_path):
        """

        Save the fund library to a h5 dataset

        Parameters
        ----------
        file_path

        Returns
        -------

        """
        hf = h5py.File(file_path, 'w')
        hf.attrs["annotations"] = str(self.annotations)
        f=hf.create_group("funds")
        for i,fund in enumerate(self.funds_set):
            grp = f.create_group("fund_{0}".format(i))
            grp.create_dataset("data",data=fund.__getstate__()["data"])
            grp["params"]=str(fund.__getstate__()["params"])
        g=hf.create_group("portfolios")
        for i,portfolio in enumerate(self.portfolios_set):
            grp = g.create_group("portfolio_{0}".format(i))
            grp.attrs["capital"]=portfolio.__getstate__()["capital"]
            grp.create_dataset("commitments",data=portfolio.__getstate__()["commitments"])
            grp.create_dataset("vintages", data=portfolio.__getstate__()["vintages"])
            funds=[hash(portfolio.funds[i]) for i in range(len(portfolio))]
            grp.create_dataset("funds", data=np.array(funds,dtype=h5py.string_dtype(encoding='utf-8')))
        hf.close()


    @classmethod
    def load(cls,lib_file):
        with open(lib_file,"rb") as fd:
             return pickle.load(fd)
         
    def load_library(self,h5_file):

        """

        Load a fund collection from a h5 dataset

        Parameters
        ----------
        h5_file

        Returns
        -------

        """
        hf = h5py.File(h5_file, 'r')
        annotations = hf.attrs.get("annotations",None)
        if annotations is not None:
            self.annotations = eval(annotations)
        fd = hf['funds']
        for key in fd.keys():
            f = Fund()
            f.load_from_numpy(np.array(fd[key]["data"],dtype=np.float64),ast.literal_eval(fd[key]["params"][()]))
            #  # Fix uncalled capital computed by T.Meyer
            #  # To be fixed by reloading the data
            #  f.fund_data[6,:] = 1.0 - np.cumsum(f.fund_data[4,:])
            #  # To be fixed also by reloading
            #  f.fund_data = np.vstack((f.fund_data,np.cumsum(f.fund_data[4,:])))
            #  f.fund_data = np.vstack((f.fund_data,np.cumsum(f.fund_data[5,:])))
            self.funds_set.append(f)
        gd = hf['portfolios']
        for key in gd.keys():
            p = Portfolio()
            state={"commitments":list(gd[key]["commitments"]),
                   "vintages":list(gd[key]["vintages"]),
                   "capital": gd[key].attrs["capital"]}
            funds=[]
            #fund_hash = np.array(gd[key]["funds"],dtype=h5py.string_dtype(encoding='utf-8'))
            fund_hash = np.array(gd[key]["funds"])
            for k in range(len(fund_hash)):
                h = int(fund_hash[k])
                fund = self.funds_set.map.get(h)
                funds.append(fund)
            state["funds"]=funds
            p.__setstate__(state)
            self.portfolios_set.append(p)
        hf.close()
        # Generate ESG cumulative weights
        self.esg_cum_weights = self.generate_esg_cum_weights()

    def __getitem__(self, index):
        """

        Get a fund from index

        Parameters
        ----------
        index: An integer value in the range

        Returns
        -------

        """
        assert(isinstance(index,int)), "Index should be an integer value"
        assert(0 <= index < len(self.set)), " Index out of bounds"
        return self.set[index]

    def __setitem__(self, index, value):
        """

        Set fund at index


        Parameters
        ----------
        index: An integer value in teh range
        value: A fund

        Returns
        -------

        None

        """
        assert(isinstance(index,int)), "Index should be an integer value"
        assert(0 <= index < len(self.set)), " Index out of bounds"
        assert (isinstance(index, Fund)), "Value should be a fund"
        assert (value.is_valid()), "Fund should contain data"
        self.set[index] = value

    def __len__(self):
        """

        Provide the size of the library

        Parameters
        ----------

        Returns
        -------
         int
          The size of the library

        """
        return len(self.set)

    def __iter__(self):
        return self

    def __next__(self):
        self.__i__ += 1
        if self.__i__ < len(self.set):
            return self.set[self.__i__]
        self.__i__=-1
        raise StopIteration


    def generate_esg_cum_weights(self):
        esg_values = []
        for i in range(len(self.funds_set)):
                fund = self.funds_set[i]
                esg = fund.get_param("ESG")
                if esg is None:
                    return None
                esg_values.append(esg)
        return np.cumsum(to_proba_scores(esg_values))
                



    def funds_selection(self,size,with_esg=True):
        """

        Select A fund/portoflio sample

        Parameters
        ----------
        size: Sample size
        replace: Replacement or not
        p: probability distribution over the fund in the library

        Returns
        -------

        """
        if not with_esg or self.esg_cum_weights is None:
            return random.choices(self.funds_set, k=size)
        return random.choices(self.funds_set, cum_weights=self.esg_cum_weights, k=size)



    def get_funds_statistics(self,indicator):
        """

        Provide some descriptive statistics for a particular feature of the fund

        Parameters
        ----------
        indicator

        Returns
        -------

        """
        data = []
        for fund in self.funds_set:
            try:
                value = fund.get_param(indicator)
                if not isinstance(value,int) and not isinstance(value, float):
                    raise Exception("Feature: {0} is not a numeric value".format(indicator))
                data.append(value)
            except KeyError as e:
                print("Error the feature {0} is unknown".format(e))
        return describe(data)


    def plot_fraction_of_total_commitment(self,t):
           t=48 # Just for the paper fix t=48
           data = np.zeros((3,t),dtype=np.float64)
           tvpis = []
           for i in range(len(self.funds_set)):
               fund = self.funds_set[i]
               if fund.get_param('LIFETIME') == t:
                    data += self.funds_set[i].fund_data[[3,7,8],:]
                    tvpis.append(fund.get_param('TVPI'))
           data = data / len(tvpis)
           xticks = np.arange(0, t+4,step=4)
           plt.xticks(xticks)
           p1, = plt.plot(data[0,:],"b-")
           p2, = plt.plot(data[1,:],"r--")
           p3, = plt.plot(data[2, :],"g:")
           ax= plt.gca()
           xticklabels = [ "{0}".format(i) for i in range(len(xticks))]
           ax.set_xticklabels(xticklabels)
           ax.set_ylim(0,2.2)
           ax.set_xlim(0,52)
           plt.legend((p1, p2, p3), ("Net Value","Cumulated contributions", "Cumulated distributions"), loc="best")
           plt.title("Fraction of total commitment")




    def select_funds_from_distribution(self,key,mean,std,N,save=None):
        try:
            close2One=1-(1e-8)
            mini,maxi=norm.interval(close2One,mean,std)
            sorted_set = sorted(self.set,key=lambda fund:fund.get_param(key))
            if mini < sorted_set[0].get_param(key) and maxi > sorted_set[-1].get_param(key):
                print("WARNING: adjust your std for sampling")
            start,end=bisect_left(sorted_set,mini,key=lambda fund:fund.get_param(key)),bisect_right(sorted_set,maxi,key=lambda fund:fund.get_param(key))
            adjusted_set = sorted_set[start:end]
            cdf = norm.cdf([adjusted_set[i].get_param(key) for i in range(len(adjusted_set))],mean,std)
            cdf = np.append(cdf,1.0)
            proba = np.diff(cdf)
            if (1.0 - np.sum(proba)) > 1e-2:
                print("WARNING: ajust your std")
            print("Sum {0}".format(np.sum(proba)))
            indexes = np.random.choice(len(adjusted_set),N,replace=False,p=proba/np.sum(proba))
            lib=Library()
            list_values=[]
            for i in indexes:
                lib.append(adjusted_set[i])
                list_values.append(adjusted_set[i].get_param(key))
            print("Mean: {0} ; std: {1}".format(np.mean(list_values),np.std(list_values)))
            if save is not None:
                lib.save(save)
            return lib
        except KeyError as e:
            print("Error the feature {0} is unknown".format(e),file=sys.stderr)
            return Library()


    def get_list_copy_of_set(self):
        list_copy = []
        for i in range(len(self.set)):
            list_copy.append(self.set[i].clone())
        return list_copy

    def split_funds(self,key,N):
        print("Splitting procedure with GaussianMixture (ncomponents={0})".format(N))
        libs=[]
        try:
            X=np.array([fund.get_param(key) for fund in self.funds_set]).reshape((-1,1))
        except KeyError as e:
            print("Error the feature {0} is unknown".format(e), file=sys.stderr)
            return []
        gm=GaussianMixture(n_components=N).fit(X)
        Y=gm.predict(X)
        for i in range(N):
            indexes,=np.where(Y==i)
            lib=Library()
            for k in indexes:
                lib.append(self.funds_set[k])
            libs.append(lib)
        return libs
