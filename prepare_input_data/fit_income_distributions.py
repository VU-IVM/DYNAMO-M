import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from numpy.random import Generator, MT19937, PCG64
import pandas as pd
import os

# write optimizer funcion
def fit_lognormal(mean_median_ratio, average_annual_income, target):
    
    # create random number generator with fixed seed
    number_generator = Generator(MT19937(seed=14))

    # init params distribution
    median_income = average_annual_income/ mean_median_ratio
    mu = np.log(median_income)
    sd = np.sqrt(2 * np.log(average_annual_income / median_income))

    
    # create income distribution
    income_distribution = np.sort(
    number_generator.lognormal(
        mu, sd, 15_000).astype(
        np.int32)) 

    start_idx = 0
    end_idx = 10
    income_deciles = np.full(10, -1, np.float32)
    for i in range(10):
        arr = np.arange(start_idx, end_idx)
        income_deciles[i] = np.percentile(income_distribution, arr).sum()
        start_idx += 10
        end_idx += 10
    # get relative contribution income percentiles
    contribution_deciles = income_deciles/ income_deciles.sum() * 100
    # score summed squared residuals
    score = np.sum((contribution_deciles - target)**2)
    # print(score)
    return score


def create_mean_to_median_ratios_raw(datafolder = 'DataDrive'):
    # read income data
    fp = os.path.join(datafolder, 'ECONOMY', 'adjusted_net_national_income_pc_WoldBank_filled.csv')
    income_db = pd.read_csv(fp)[['Country Code', '2015 [YR2015]']].dropna()
    
    # load distribution data
    cols_to_keep = [f'd{decile}' for decile in np.arange(1, 11)]
    cols_to_keep.append('year')
    # fp = os.path.join(datafolder, 'ECONOMY', 'wiid-data.xlsx')
    fp = os.path.join(datafolder, 'ECONOMY', 'WIID_28NOV2023.xlsx')
    distribution_data = pd.read_excel(fp, sheet_name='Sheet1', index_col='c3')[cols_to_keep].dropna()
    # columns to keep

    # iniate export dictionary
    export = {}

    # iterate over rows and fit 
    for _, country in income_db.iterrows():
        iso_code = country['Country Code'] 
        if iso_code in distribution_data.index and country['2015 [YR2015]'] != '..' :
            print(f'Estimating {iso_code}')
            # inii
            export[iso_code] = {}
            # mean income
            average_annual_income = float(country['2015 [YR2015]'])
            # get targets 
            data = distribution_data.loc[iso_code]
            if data.ndim == 2:
                data = data.iloc[-1]
            year = data['year']
            target = np.array(data.drop('year'))

            # optimize
            args = (average_annual_income, target)
            x0 = 1
            result = minimize(fit_lognormal, x0, method='nelder-mead', args=args).x[0]
            export[iso_code]['mean_income'] = average_annual_income
            export[iso_code]['median_income'] = average_annual_income/ result
            export[iso_code]['mean_to_mean_ratio'] = result
            export[iso_code]['year'] = year


    export_pandas = pd.DataFrame(export).transpose()
    export_pandas.to_csv(os.path.join(datafolder, 'ECONOMY', 'PROCESSED', 'mean_to_median_income.csv'))

def create_mean_to_median_ratios(datafolder = 'DataDrive'):
    # read income data
    fp = os.path.join(datafolder, 'ECONOMY', 'adjusted_net_national_income_pc_WoldBank.csv')
    income_db = pd.read_csv(fp)[['Country Code', '2015 [YR2015]']].dropna()
    
    # load distribution data
    fp = os.path.join(datafolder, 'ECONOMY', 'wiid-data.xlsx')
    distribution_data = pd.read_excel(fp, sheet_name=0, index_col='ISO')

    # iniate export dictionary
    export = {}

    # iterate over rows and fit 
    for _, country in income_db.iterrows():
        iso_code = country['Country Code'] 
        if iso_code in distribution_data.index and country['2015 [YR2015]'] != '..' :
            print(f'Estimating {iso_code}')
            # inii
            export[iso_code] = {}
            # mean income
            average_annual_income = float(country['2015 [YR2015]'])
            # get targets 
            data = distribution_data.loc[iso_code]
            data = data.drop(['YEAR', 'COUNTRY'])
            target = np.array(data)

            # optimize
            args = (average_annual_income, target)
            x0 = 1
            result = minimize(fit_lognormal, x0, method='nelder-mead', args=args).x[0]
            export[iso_code]['mean_income'] = average_annual_income
            export[iso_code]['median_income'] = average_annual_income/ result
            export[iso_code]['mean_median_ratio'] = result

    export_pandas = pd.DataFrame(export).transpose()
    export_pandas.to_csv(os.path.join(datafolder, 'ECONOMY', 'PROCESSED', 'mean_median_WIID.csv'))


if __name__ == '__main__':
    create_mean_to_median_ratios()

