from stairs.libraries import Library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse
import logging
sns.set_theme(style="darkgrid")
np.seterr(divide='ignore', invalid='ignore')
logging.basicConfig(level=logging.INFO,format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
MAX_COLS=2

plt.rcParams.update({"text.usetex": True,'axes.titlesize':38})
plt.rcParams.update({'font.size': 22,'legend.fontsize': 20,
         'axes.labelsize': 'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'})
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


MAP = {"ID":"Investment Degrees","NAV":"Net Asset Values","Cash":"Portfolio Cash","Commit":"Commitments","Distrib":"Distributions","Contrib":"Contributions","ESG":"Distribution of ESG scores"}

class Stats:

    def __init__(self,lib_portfolios):
        self.lib_portfolios = []
        self.heuristics = []
        self.dfs = []
        self.funds_dfs = []
        logging.info("Start computing max portfolios lifetime")
        self.max_lifetime,self.max_funds=self._compute_max_lifetime_max_funds(lib_portfolios)
        logging.info("Done")
        logging.info("Generate DataFrames")
        for k,lib in enumerate(lib_portfolios):
            self.lib_portfolios.append(lib)
            try:
                self.heuristics.append(r'${0}$'.format(lib.get_annotation('heuristic')))
                #self.heuristics.append('heuristic{0}'.format(k))
            except KeyError:
                self.heuristics.append('heuristic{0}'.format(k))
            self.dfs.append(self._build_dataframe(lib))
            self.funds_dfs.append(self._build_funds_dataframe(lib))
        logging.info("Dataframes created")
        #self.df = pd.concat(dfs, keys=self.heuristics)
        #with pd.ExcelWriter("test.xlsx") as writer:
        #     self.df.to_excel(writer,float_format="%.4f")

    def _compute_max_lifetime_max_funds(self,lib_portfolios):
        max_lifetime = 0
        max_funds = 0
        for lib in lib_portfolios:
             if not lib.arePortfoliosSelected():
                lib.switch()
             for portfolio in lib:
                 lt_portfolio = portfolio.get_portfolio_lifetime()
                 nb_funds = len(portfolio.funds)
                 if lt_portfolio > max_lifetime:
                     max_lifetime = lt_portfolio
                 if nb_funds > max_funds:
                     max_funds = nb_funds
        return max_lifetime,max_funds

    def _align2max_lifetime(self,np_array,value=0):
        return np.pad(np_array,(0,self.max_lifetime - len(np_array)),'constant', constant_values=value)

    
    def _build_dataframe(self,lib):
        dict_data = dict()
        dict_data["quarters"] = np.arange(self.max_lifetime)
        for k,portfolio in enumerate(lib):
            ID = portfolio.get_ID(i=-1)
            dict_data["ID_{0}".format(k)] = self._align2max_lifetime(ID)
            dict_data["Cash_{0}".format(k)] = self._align2max_lifetime(portfolio.get_cash(i=-1))
            dict_data["NAV_{0}".format(k)] = self._align2max_lifetime(portfolio.get_net_asset_value(i=-1))
            dict_data["Contrib_{0}".format(k)] = self._align2max_lifetime(portfolio.get_contributions(i=-1))
            dict_data["Distrib_{0}".format(k)] = self._align2max_lifetime(portfolio.get_distributions(i=-1))
            dict_data["Commit_{0}".format(k)] = self._align2max_lifetime(self._build_commitments(portfolio))
            dict_data["Shortage_{0}".format(k)] = self._build_shortage(ID)

            dict_data["Frac_commit_{0}".format(k)] = dict_data["Commit_{0}".format(k)]/ (dict_data["Cash_{0}".format(k)] + dict_data["NAV_{0}".format(k)])
            dict_data["Frac_contrib_{0}".format(k)] = dict_data["Contrib_{0}".format(k)]/ (dict_data["Cash_{0}".format(k)] + dict_data["NAV_{0}".format(k)])
            dict_data["Frac_distrib_{0}".format(k)] = dict_data["Distrib_{0}".format(k)]/ (dict_data["Cash_{0}".format(k)] + dict_data["NAV_{0}".format(k)])

            
        df = pd.DataFrame(dict_data)
        df.set_index("quarters")
        #with pd.ExcelWriter("test.xlsx") as writer:
        #     df.to_excel(writer,float_format="%.4f")
        return df


    def _build_funds_dataframe(self,lib):
        dict_data = dict()
        for k,portfolio in enumerate(lib):
            dict_data["ESG_{0}".format(k)] = [float(fund.get_param("ESG")) for fund in portfolio.funds] + [np.nan] * (self.max_funds - len(portfolio.funds))
        df = pd.DataFrame(dict_data)
        return df

    

    def _build_commitments(self,portfolio):
        df = pd.DataFrame({'commitments':portfolio.get_commitments(),'vintages':portfolio.get_vintages()})
        df = df.groupby(['vintages']).agg(sum)
        return df['commitments'].to_numpy()


    def _build_shortage(self,ID):
        index = np.argmax(ID > 1.0)
        shortage = np.zeros_like(ID)
        if index > 0:
            shortage[index:] = 1
            return self._align2max_lifetime(shortage,value=1)
        else:
            return self._align2max_lifetime(shortage)


    def table(self):
        df_summary={}
        for k,df in enumerate(self.dfs):
            df_shortage = df.filter(regex=("Shortage.*"))
            N_portfolios = df_shortage.shape[-1]
            df_mean = df_shortage.mean(axis=1)
            df1 = df.filter(regex=("{0}.*".format("ID")))
            stats_df = df1.apply(pd.DataFrame.describe,axis=1)[["mean","std"]]
            #stats_df["proba"]=df1.apply(lambda x: x[x>1.0].count(),axis=1)/10
            # Make diff explicitely
            stats_df["proba"]=df_mean*100
            df_summary[self.heuristics[k]] = stats_df
        df_summary = pd.concat(df_summary,axis=1)
        df_summary["years"] = df_summary.index // 4
        df_summary.set_index("years")
        df_summary = df_summary.groupby("years").mean().round(2)
        df_summary.to_excel("/tmp/output.xlsx")
        df_summary.to_latex("/tmp/output.tex")



    def plot_mean(self,key,name):
        cm = plt.get_cmap('gist_rainbow')
        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 10.5)
        xticks = np.arange(0, self.max_lifetime,step=8)
        xticklabels = [ "{0}".format(2*i) for i in range(len(xticks))]
        ax.set_xticklabels(xticklabels)
        if key == "ID":
           ax.set_ylim(0,1.2)
        # For the SSCI paper, comment afterwards
        heuristics=["$DZ^{1}(t)$","$DZ^{2}(t)$","$DZ^{3}(t)$","$S^{best}(t)$"]
        for k,df in enumerate(self.dfs):
            df1 = df.filter(regex=("{0}.*".format(key)))
            #df1.plot(title="Investment Degree",legend=False,xticks=xticks,ax=ax)
            df1.mean(axis=1).plot(linewidth=5,legend=True,label=heuristics[k],ax=ax,xticks=xticks)
        plt.xlabel("Years")
        #plt.title(name,fontsize=24)
        plt.tight_layout()
        #plt.show()
        plt.savefig("/home/manu/Desktop/compare.pdf",format="pdf")


    def plot_all(self,index,key,name):
        cm = plt.get_cmap('gist_rainbow')
        if index >= 0 and index < len(self.dfs):
            dfs = [self.dfs[index]]
        else:
            dfs = self.dfs
        N = len(dfs)
        nrows = int(np.ceil(N/MAX_COLS))
        ncols = 2
        fig, ax = plt.subplots(nrows=nrows,ncols=min(N,ncols))
        fig.set_size_inches(18.5, 10.5)
        if N > 1:
            ax = ax.flatten()
            for i in range(N,nrows*ncols):
                fig.delaxes(ax[i])
        else:
            ax = [ax]
        xticks = np.arange(0, self.max_lifetime,step=8)
        xticklabels = [ "{0}".format(2*i) for i in range(len(xticks))]
        #fig.suptitle(name, fontsize=28)
        for k,df in enumerate(dfs):
            ax[k].set_xticklabels(xticklabels)
            df1 = df.filter(regex=("{0}.*".format(key)))
            df1.plot(legend=False,xticks=xticks,ax=ax[k])
            df1.mean(axis=1).plot(color="red",linewidth=5,legend=True,label="Average Investment Degree",ax=ax[k],xticks=xticks)
            #ax[k].set_title(self.heuristics[k],fontsize=28)
            if key == "ID":
                ax[k].set_ylim(0,1.2)
        plt.xlabel("Years")
        plt.tight_layout()
        #plt.show()
        plt.savefig("/home/manu/Desktop/screenshot_xp5_15.pdf",format="pdf")


    def plot_ci(self,index,key,name):
        cm = plt.get_cmap('gist_rainbow')
        if index >= 0 and index < len(self.dfs):
            dfs = [self.dfs[index]]
            heuristics = [self.heuristics[index]]
        else:
            dfs = self.dfs
            heuristics = self.heuristics
        N = len(dfs)
        nrows = int(np.ceil(N/MAX_COLS))
        ncols = 2
        fig, ax = plt.subplots(nrows=nrows,ncols=min(N,ncols))
        fig.set_size_inches(18.5, 10.5)
        if N > 1:
            ax = ax.flatten()
            for i in range(N,nrows*ncols):
                fig.delaxes(ax[i])
        else:
            ax = [ax]
        xticks = np.arange(0, self.max_lifetime,step=8)
        xticklabels = [ "{0}".format(2*i) for i in range(len(xticks))]
        #fig.suptitle(name, fontsize=14)
        for k,df in enumerate(dfs):
            ax[k].set_xticklabels(xticklabels)
            df1_short = df.filter(regex=("Shortage.*")).transpose().sum(axis=1).to_numpy()
            indexes_ok = np.where(df1_short<1.0) [0]
            indexes_nok = np.where(df1_short>1.0) [0]
            columns_ok = map(lambda x: "ID_{0}".format(x),indexes_ok.tolist())
            df1 = df[columns_ok]
            av = df1.mean(axis=1)
            ci = 1.96 * df1.std(axis=1)
            #ax[k].fill_between(df1.index, (av-ci), (av+ci), color='b', alpha=.2)
            ax[k].fill_between(df1.index, df1.min(axis=1), df1.max(axis=1), color='b', alpha=.2)
            df1.mean(axis=1).plot(color="red",linewidth=5,legend=True,label="Average Investment Degree",ax=ax[k],xticks=xticks)
            #ax[k].set_title(heuristics[k],fontsize='x-large')
            if key == "ID":
                ax[k].set_ylim(0,1.2)
                
                df_shortage = df.filter(regex=("Shortage.*"))
                N_portfolios = df_shortage.shape[-1]
                df_mean = df_shortage.mean(axis=1)
                df_mean = df_mean.mask(df_mean.index %4 != 0)
                df_mean.plot.bar(rot=0,linewidth=2,color="black",legend=True,label="Percentage of overinvested portfolios",ax=ax[k],xticks=xticks)
                for l,p in enumerate(ax[k].patches):
                    if l%4 == 0 and df_mean[l] > 0:
                       ax[k].annotate("{0}".format(np.round(df_mean[l]*100,1)), (p.get_x() + p.get_width()/2, p.get_height() * 1.1),ha='center',va='center',xytext=(0, 10), textcoords='offset points')
        plt.xlabel("Years")
        plt.tight_layout()
        #plt.show()
        plt.savefig("/home/manu/Desktop/screenshot_xp5_15.pdf",format="pdf")


    def plot_frac_total_assets(self,index=-1):
        cm = plt.get_cmap('gist_rainbow')
        if index >= 0 and index < len(self.dfs):
            dfs = [self.dfs[index]]
            heuristics = [self.heuristics[index]]
        else:
            dfs = self.dfs
            heuristics = self.heuristics
        N = len(dfs)
        nrows = int(np.ceil(N/MAX_COLS))
        ncols = 2
        fig, ax = plt.subplots(nrows=nrows,ncols=min(N,ncols))
        if N > 1:
            ax = ax.flatten()
            for i in range(N,nrows*ncols):
                fig.delaxes(ax[i])
        else:
            ax = [ax]
        xticks = np.arange(0, self.max_lifetime,step=8)
        xticklabels = [ "{0}".format(2*i) for i in range(len(xticks))]
        fig.suptitle("Fraction of total assets", fontsize=38)
        for k,df in enumerate(dfs):
            ax[k].set_xticklabels(xticklabels)
            df.filter(regex=("Frac_commit.*")).mean(axis=1).plot(linewidth=2,color="black",legend=True,label="Commitments",ax=ax[k],xticks=xticks)
            df.filter(regex=("Frac_contrib.*")).mean(axis=1).plot(linewidth=2,color="red",legend=True,label="Contributions",ax=ax[k],xticks=xticks)
            df.filter(regex=("Frac_distrib.*")).mean(axis=1).plot(linewidth=2,color="green",legend=True,label="Distributions",ax=ax[k],xticks=xticks)
            ax[k].set_title(heuristics[k])
        plt.xlabel("Years")
        #plt.tight_layout()
        plt.show()


    def plot_shortage(self,index=-1):
        cm = plt.get_cmap('gist_rainbow')
        if index >= 0 and index < len(self.dfs):
            dfs = [self.dfs[index]]
            heuristics = [self.heuristics[index]]
        else:
            dfs = self.dfs
            heuristics = self.heuristics
        N = len(dfs)
        nrows = int(np.ceil(N/MAX_COLS))
        ncols = 2
        fig, ax = plt.subplots(nrows=nrows,ncols=min(N,ncols))
        if N > 1:
            ax = ax.flatten()
            for i in range(N,nrows*ncols):
                fig.delaxes(ax[i])
        else:
             ax=[ax]
        fig.suptitle("Cumulated distribution of overinvested portfolios", fontsize=38)
        for k,df in enumerate(dfs):
            df_shortage = df.filter(regex=("Shortage.*"))
            N_portfolios = df_shortage.shape[-1]
            df_mean = df_shortage.mean(axis=1)
            df_year = df_mean.groupby(df_mean.index // 4).max() * 100 
            df_year.plot.bar(rot=0,linewidth=2,color="black",legend=True,label="Percentage of overinvested portoflios",ax=ax[k])
            for l,p in enumerate(ax[k].patches):
                ax[k].annotate("{0}".format(np.round(df_year[l],1)), (p.get_x() + p.get_width()/2, p.get_height() * 1.1),ha='center',va='center',xytext=(0, 10), textcoords='offset points')
            ax[k].set_ylim(0,100)
            ax[k].set_title(heuristics[k])
        plt.xlabel("Years")
        plt.show()


    def plot_esg_distributions(self):
        data_dict=dict()
        for k, df in enumerate(self.funds_dfs):
            data_dict[self.heuristics[k]] = df.to_numpy().flatten()
        df_esg = pd.DataFrame(data_dict).stack().reset_index().drop(columns="level_0",axis=1).rename(columns={'level_1': 'Heuristic', 0: 'ESG scores'})
        sns.displot(data = df_esg, x="ESG scores",hue="Heuristic", kde=True)
        plt.title("Distribution of ESG scores")
        plt.show()





        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(nargs="+", dest="paths", help="One or more simulated portfolios in h5 format")
    parser.add_argument("-m","--metric", choices=["ID","NAV","Cash","Commit","Distrib","Contrib","Fraction","Shortage","ESG"], help="Choose metric to dsiplay")
    parser.add_argument("-c","--compare", action='store_true', default=False, help="Gather all in one")
    parser.add_argument("-ci","--confidence", action='store_true',default=False, help="Plot rather a confidence interval")
    parser.add_argument("-t","--table", action='store_true',default=False, help="Output comparison table")
    parser.add_argument("-i","--index", type=int, default=-1 ,help="Display single simulation")
    args = parser.parse_args()
    libs = []
    logging.info("Start loading portfolio libraries")
    for path in args.paths:
        lib = Library.load(path)
        logging.info("----> Loading: {0}".format(path)) 
        libs.append(lib)
    stats = Stats(libs)
    if args.table:
        stats.table()
    if args.metric == "Fraction":
       stats.plot_frac_total_assets(args.index)
    elif args.metric == "Shortage":
       stats.plot_shortage(args.index)
    elif args.metric == "ESG":
        stats.plot_esg_distributions()
    elif args.metric:
        if args.compare:
            stats.plot_mean(args.metric,MAP[args.metric])
        else:
            if args.confidence:
                stats.plot_ci(args.index,args.metric,MAP[args.metric])
            else:
                stats.plot_all(args.index,args.metric,MAP[args.metric])


#    #stats.plot_mean("Cash","Remaining cash")
#    #stats.plot_all("F","Investment degrees")
#    #stats.plot_ci("F","Investment degrees")
#    #stats.plot_frac_total_assets()
#    #stats.plot_shortage()
#    stats.plot(0,"ID","Investment degree")
#




    





