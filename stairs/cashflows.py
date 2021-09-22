from stairs.modified_yale_model_v10 import *
from stairs.fund import Fund
from stairs.libraries import Library

from stairs.utils import read_file,read_parameters_file
import math



class CashFlowGenerator:



    def load_cashflow_lib(self,h5_path,**kwargs):

        file_draw_downs = kwargs.get("file_draw_downs","")
        file_mgmt_fees = kwargs.get("file_mgmt_fees","")
        file_repayments = kwargs.get("file_repayments","")
        file_fixed_returns = kwargs.get("file_fixed_returns","")
        file_values = kwargs.get("file_values","")
        uncalled_capital = kwargs.get("uncalled_capital","")

        file_parameters = kwargs.get("file_parameters","")

        generators=[ read_file(file_draw_downs,sep=";",cast=float),
                     read_file(file_mgmt_fees,sep=";",cast=float),
                     read_file(file_repayments, sep=";", cast=float),
                     read_file(file_fixed_returns, sep=";", cast=float),
                     read_file(file_values, sep=";", cast=float),
                     read_file(uncalled_capital, sep=";", cast=float),
                     read_parameters_file(file_parameters, sep=";")]


        lib=Library()
        cnt=1
        while True:
            print("Importing fund {0}".format(cnt))

            try:
                all_data=[next(gen) for gen in generators]

                lifetime = all_data[-1]['LIFETIME']
                f=Fund()
                f.load_from_generator(draw_downs=np.array(all_data[0][:lifetime],dtype=np.float32),
                                      mgmt_fees=np.array(all_data[1][:lifetime],dtype=np.float32),
                                      repayments=np.array(all_data[2][:lifetime],dtype=np.float32),
                                      fixed_returns=np.array(all_data[3][:lifetime],dtype=np.float32),
                                      values=np.array(all_data[4][:lifetime],dtype=np.float32),
                                      uncalled_capital=np.array(all_data[5][:lifetime],dtype=np.float32),
                                      parameters=all_data[-1]
                                      )
                lib.append(f)

            except StopIteration as err:
                   print(err.value)
                   break
            cnt+=1

        shuffle = kwargs.get("shuffle",False)
        rev = kwargs.get("reversed",False)
        size= kwargs.get("size",None)

        if shuffle:
            lib.shuffle()
        if rev:
            lib.rev()

        if size is not None:
            # Maybe create a specific method for this a the class
            lib.funds_set = lib.funds_set[:size]
        lib.save(h5_path)








    def generate_cashflow_lib(self,file_path,**kwargs):
        funds_cashflows = self(**kwargs)
        sample_draw_downs = funds_cashflows[0]
        sample_mgmt_fees = funds_cashflows[1]
        sample_repayments = funds_cashflows[2]
        sample_values = funds_cashflows[3]

        assert(sample_draw_downs.shape ==
               sample_mgmt_fees.shape ==
               sample_repayments.shape ==
               sample_values.shape)

        lib=Library()
        for i in range(sample_draw_downs.shape[0]):
            f=Fund()
            f.load_from_generator(draw_downs=sample_draw_downs[i,:],
                   mgmt_fees=sample_mgmt_fees[i,:],
                   repayements=sample_repayments[i,:],
                   values=sample_values[i,:])
            lib.append(f)
        lib.save(file_path)

    def __call__(self,**kwargs):
        return self.call_modified_yale_model_v10(**kwargs)

    def call_modified_yale_model_v10(self,**kwargs):

        cashflow_samples = kwargs.get("cashflow_samples",100000)
        yale_growth = kwargs.get("yale_growth",0.15)
        yale_years = kwargs.get("yale_years",10)
        yale_strategy = kwargs.get("yale_strategy",'GENERAL')
        periodicity_type = kwargs.get("periodicity_type",'YEARLY')
        harvest = kwargs.get("harvest", False)
        tolerance_irr = kwargs.get("tolearnce_irr",0.01)
        tolerance_tvpi = kwargs.get("tolerance_tvpi",0.2)

        # *** Probability distribution assumed for repayments
        # *** TRIANGULAR is assumed for draw-downs. However, TRIANGULAR is
        # *** problematic for repayments as it tends to overestimate the likelihood
        # *** of above average positive cash-flows. EXPONENTIAL solves this issue
        # random_type = 'TRIANGULAR'
        random_type = kwargs.get("random_type",'EXPONENTIAL')
        precision = kwargs.get("precision", 0.0000000000001)

        # ***********************************************************************
        # ***********************************************************************
        # ***                    START OF CORE LOGIC                          ***
        # ***********************************************************************
        # ***********************************************************************

        # *** Initialisation: get period conversion parameters
        # *** (e.g. to quarterly / biannually)
        convert_period = set_period_conversion()

        # ***********************************************************************
        # *** Initialize Yale-Model
        # ***********************************************************************

        yale_param, yale_cntrb_rate = \
            set_yale_parameters(yale_strategy, yale_growth, yale_years)

        # ***********************************************************************
        # *** First step: build plain-vanilla Yale projection, for yearly
        # *** expectations
        # ***********************************************************************

        yale_draw_downs, yale_repayments, yale_values, yale_uncalled_capital \
            = build_yale(yale_param, yale_cntrb_rate)

        period_factor = 1  # *** Yale model based on yearly projection

        # *** Check validity of first step result
        check_yale_result(yale_param, yale_draw_downs,
                          yale_repayments, yale_years,
                          period_factor, check_show_yale)


        # *** For checking and comparing overall results later
        yale_tvpi = sum(yale_repayments) / sum(yale_draw_downs)
        yale_in = sum(yale_draw_downs)
        yale_out = sum(yale_repayments)

        # ***********************************************************************
        # *** Initialize Modified Yale-model
        # ***********************************************************************

        mod_yale_param = set_modified_yale_parameters(yale_strategy,
                                                      yale_years)

        standard_draw_down_size = modified_yale_cashflow_size(yale_draw_downs,
                                                              mod_yale_param,
                                                              'CONTRIBUTIONS',
                                                              random_type)
        frequency_draw_down_band = \
            modified_yale_frequency_band(yale_years, mod_yale_param,
                                         'CONTRIBUTIONS')

        standard_repayment_size = modified_yale_cashflow_size(yale_repayments,
                                                              mod_yale_param,
                                                              'DISTRIBUTIONS',
                                                              random_type)
        frequency_repayment_band = \
            modified_yale_frequency_band(yale_years, mod_yale_param,
                                         'DISTRIBUTIONS')

        # ***********************************************************************
        # *** Generate randomized samples based on Modified-Yale-Model
        # ***********************************************************************

        # *** Number of cash-flows and dimensioning of result structure depend
        # *** on targeted period
        period_factor = period_transform('MONTHLY',
                                         periodicity_type,
                                         convert_period)
        period_factor = int(12 * period_factor)
        sample_dimension = period_factor * yale_years

        # *** IRRs need to be converted to targeted periodicity. This step is
        # *** not necessary for TVPIs as they are indifferent to fund's lifetime
        target_irr = growth_rate_transform(yale_growth, period_factor)
        tolerance_irr = growth_rate_transform(tolerance_irr, period_factor)

        # *** Matrix for samples to be generated as results of routine
        sample_draw_downs = np.zeros((cashflow_samples,
                                      sample_dimension))
        sample_mgmt_fees = np.zeros((cashflow_samples,
                                     sample_dimension))
        sample_repayments = np.zeros((cashflow_samples,
                                      sample_dimension))
        sample_values = np.zeros((cashflow_samples,
                                  sample_dimension))
        sample_uncalled_capital = np.zeros((cashflow_samples,
                                            sample_dimension))
        sample_internal_age = np.zeros((cashflow_samples,
                                        sample_dimension))

        # *** For checking and comparing results against original Yale-Model.
        # *** We need to hold inf-flows and out-flows separately before
        # *** calculating the average TVPI in the last step
        sample_irr = np.zeros(cashflow_samples)
        sample_in_flow = np.zeros(cashflow_samples)
        sample_out_flow = np.zeros(cashflow_samples)
        sample_tvpi = np.zeros(cashflow_samples)

        sample_fail = 0
        icnt = 0
        while cashflow_samples > icnt:
            # *** Randomize Yale-Model's drawn-downs and repayments
            try_draw_downs, try_management_fees = \
                modified_yale_contributions_triangular(yale_draw_downs,
                                                       yale_param)
            if random_type == 'TRIANGULAR':
                try_repayments = \
                    modified_yale_distributions_triangular(yale_repayments,
                                                           mod_yale_param)
            elif random_type == 'EXPONENTIAL':
                try_repayments = \
                    modified_yale_distributions_exponential(yale_repayments)
            else:
                pass
            pass  # ELSE

            # *** Convert into targeted periodicity
            if periodicity_type == 'YEARLY':
                # *** For yearly cash-flows as target no further conversion
                # *** required
                pass
            else:
                # *** Otherwise convert into monthly cash-flows first
                cashflow_type = 'CONTRIBUTIONS'
                monthly_draw_downs = \
                    build_monthly_cashflows(yale_param, try_draw_downs,
                                            cashflow_type, random_type,
                                            standard_draw_down_size,
                                            frequency_draw_down_band)

                cashflow_type = 'DISTRIBUTIONS'
                monthly_repayments = \
                    build_monthly_cashflows(yale_param, try_repayments,
                                            cashflow_type, random_type,
                                            standard_repayment_size,
                                            frequency_repayment_band)

                monthly_management_fees = \
                    monthly_management_fee_schedule(yale_param,
                                                    try_management_fees)

                if periodicity_type == 'MONTHLY':
                    # *** For monthly cash-flows as target no further
                    # *** conversion required
                    try_draw_downs = monthly_draw_downs
                    try_management_fees = monthly_management_fees
                    try_repayments = monthly_repayments
                else:
                    # *** Otherwise convert monthly cash-flows into targeted
                    # *** cash-flow frequency
                    try_draw_downs = \
                        convert_to_target_period(monthly_draw_downs,
                                                 period_factor)
                    try_management_fees = \
                        convert_to_target_period(monthly_management_fees,
                                                 period_factor)
                    try_repayments = \
                        convert_to_target_period(monthly_repayments,
                                                 period_factor)

                pass  # ELSE
            pass  # ELSE

            # *** Check whether generated cash-flow samples are valid
            cf_valid, try_irr, try_in_flow, try_out_flow = \
                check_cashflow_validity(try_draw_downs,
                                        try_management_fees,
                                        try_repayments)

            try_tvpi = try_out_flow / try_in_flow

            if harvest and cf_valid:
                # *** Harvesting option: only accept valid cash-flows
                # *** samples with IRRs and TVPIs close to Yale-Model's
                # *** IRRs and TVPIs (within set tolerances)
                cf_valid = check_irr_tvpi_target_met(try_irr,
                                                     try_tvpi,
                                                     target_irr,
                                                     yale_tvpi,
                                                     tolerance_irr,
                                                     tolerance_tvpi)

                if cf_valid:
                    print(icnt, 'sample found for',
                          periodicity_type, 'cash-flow')
                    print('-> Targeted IRR', target_irr, 'found:', try_irr)
                    print('-> Targeted TVPI', yale_tvpi, 'found:', try_tvpi)
                else:
                    pass
                pass  # ELSE
            else:
                pass
            pass  # ELSE

            if cf_valid:
                sample_draw_downs[icnt, :] = try_draw_downs
                sample_mgmt_fees[icnt, :] = try_management_fees
                sample_repayments[icnt, :] = try_repayments
                sample_values[icnt, :] = \
                    cashflow_consistent_nav(try_draw_downs +
                                            try_management_fees,
                                            try_repayments)

                sample_uncalled_capital[icnt, :] = \
                    determine_uncalled_capital(try_draw_downs +
                                               try_management_fees)

                sample_internal_age[icnt, :] = \
                    determine_internal_age(try_draw_downs + try_management_fees,
                                           try_repayments,
                                           sample_values[icnt, :],
                                           sample_uncalled_capital[icnt, :])

                sample_irr[icnt] = try_irr
                sample_in_flow[icnt] = try_in_flow
                sample_out_flow[icnt] = try_out_flow
                sample_tvpi[icnt] = try_tvpi
                icnt += 1
            else:
                sample_fail += 1
            pass  # ELSE
        pass  # WHILE

        print('ACCEPTANCE RATE',
              1 - sample_fail / (cashflow_samples + sample_fail))

        # *** For displaying and comparison adjust Yale-Model's
        # *** average projections to targeted periodicity
        yale_draw_downs_adj = yale_adjust_periodicity(yale_draw_downs,
                                                      period_factor,
                                                      'CASHFLOWS')
        yale_repayments_adj = yale_adjust_periodicity(yale_repayments,
                                                      period_factor,
                                                      'CASHFLOWS')
        yale_values_adj = yale_adjust_periodicity(yale_values,
                                                  period_factor,
                                                  'VALUES')


        print('Yale IRR', yale_growth, 'Period-adjusted',
              growth_rate_transform(yale_growth, period_factor),
              'Yale TVPI', yale_tvpi)
        print('Average IRR', np.average(sample_irr),
              'Average TVPI', np.average(sample_out_flow) /
              np.average(sample_in_flow))
        print('Variance IRR', np.var(sample_irr),
              'Variance TVPI', np.var(sample_tvpi))

        return sample_draw_downs,sample_mgmt_fees, sample_repayments,sample_values

    # ***************************************************************************
    # ***************************************************************************
    # ***************************************************************************
    # ***                                                                     ***
    # ***                               END                                   ***
    # ***                                                                     ***
    # ***************************************************************************
    # ***************************************************************************




if __name__ == "__main__":
    directory="../../data"
    gen = CashFlowGenerator()
    gen.load_cashflow_lib("../../data/lib.h5",size=100,shuffle=True, file_draw_downs=os.path.join(directory,"yale_plus_draw_downs.txt"),
    #gen.load_cashflow_lib("../../data/lib_cashflows.h5", file_draw_downs=os.path.join(directory,"yale_plus_draw_downs.txt"),
                                                file_mgmt_fees=os.path.join(directory,"yale_plus_management_fees.txt"),
                                                file_repayments=os.path.join(directory,"yale_plus_repayments.txt"),
                                                file_fixed_returns=os.path.join(directory, "yale_plus_fixed_returns.txt"),
                                                file_values=os.path.join(directory, "yale_plus_market_values.txt"),
                                                uncalled_capital=os.path.join(directory, "yale_plus_uncalled_capital.txt"),
                                                file_parameters=os.path.join(directory,"yale_plus_parameters.txt"))
