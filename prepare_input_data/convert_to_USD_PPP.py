import pandas as pd
import numpy as np
import os

def create_conversion_table():
    # read floodplains in model
    agents_fn = r'/scistor/ivm/ltf200/COASTMOVE/DataDrive/SLR/households_gdl_2015/xxlarge'
    countries = np.unique([region[:3] for region in os.listdir(agents_fn)])


    # exchange rates WorldBank
    path = os.path.join('DataDrive', 'ECONOMY', 'conversion_rates', 'LCU_per_USD.csv')
    exchange_rates_worldbank = pd.read_csv(path, skiprows=4, index_col=['Country Code'])
    exchange_rates_worldbank = exchange_rates_worldbank['2015']
    dollar_to_euro = exchange_rates_worldbank.loc['EMU']

    # USD PPP conversion rates WorldBank
    path = os.path.join('DataDrive', 'ECONOMY', 'conversion_rates', 'LCU_per_USD_PPP.csv')
    exchange_rates_worldbank_ppp = pd.read_csv(path, skiprows=4, index_col=['Country Code'])
    exchange_rates_worldbank_ppp = exchange_rates_worldbank_ppp['2015']


    # construct conversion table
    conversion_table = pd.DataFrame()
    for i, country in enumerate(countries):
        # load exchange rate world bank
        if country in exchange_rates_worldbank.index:
            exchange_rate = exchange_rates_worldbank.loc[country]
            LCU_per_USD_ppp = exchange_rates_worldbank_ppp.loc[country]
            USD_ppp_per_LCU = 1/LCU_per_USD_ppp
            EUR_to_LUC  = exchange_rate/dollar_to_euro
        else: 
            exchange_rate = 'missing'
            USD_ppp_per_LCU = 'missing'

        row = {'alpha3': country, 'USD_to_LUC': exchange_rate, 'EUR_to_LUC': EUR_to_LUC, 'USD_ppp_per_LCU': USD_ppp_per_LCU}
        conversion_table = pd.concat([conversion_table, pd.DataFrame(row, index=[i])])


    # save
    target_folder = os.path.join('DataDrive', 'ECONOMY', 'conversion_rates')
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    conversion_table.to_csv(os.path.join(target_folder, 'conversion_rates.csv'))

if __name__ == '__main__':
    create_conversion_table()