import pandas as pd
import os 
import numpy as np

def create_scaled_df(out, source_cost, source_country):
    # load income 2015
    path = os.path.join('DataDrive', 'ECONOMY', 'adjusted_net_national_income_pc_WoldBank_filled.csv')
    
    # load and clean income sheet World Bank
    df = pd.read_csv(path)[['Country Code', '2015 [YR2015]']].set_index('Country Code', drop=True).dropna()
    df.drop(df[df['2015 [YR2015]'] == '..'].index, inplace = True)
    df['2015 [YR2015]'] = np.array(df['2015 [YR2015]'], np.float32)

    # scale for other countries based on ratio income/ fixed cost
    # income_source_country
    income_source_country = df.loc[source_country]['2015 [YR2015]']
    # get ratio
    ratio = source_cost/ income_source_country

    # scale for all countries in df
    countries = []
    scaled_costs = []

    for iso3_code, row in df.iterrows():
        
        income_country = row['2015 [YR2015]']
        scaled_costs.append(int(income_country * ratio))
        countries.append(iso3_code)
    
    export_df = pd.DataFrame({'iso3_code': countries, 'fixed_cost_scaled': scaled_costs}).set_index('iso3_code', drop=True)
    
    # target folder
    export_fn = os.path.join('DataDrive', 'ECONOMY', 'PROCESSED')
    if not os.path.exists(export_fn):
        os.makedirs(export_fn)
    export_df.to_csv(os.path.join(export_fn, out))

def scale_dike_elevation_cost_based_on_construction_index(out = 'scaled_dike_elevation_cost.csv', source_cost_EU = 6.17E6):
    # load construction price index from Tiggeloven
    path = os.path.join('DataDrive', 'ECONOMY', 'conversion_rates', 'PPP_MER_Con_G109.csv')
    df = pd.read_csv(path, index_col='ISO')
    scaled_costs = []
    countries = []
    for iso3_code, row in df.iterrows():
        cc_index = row['constructioncost_index']
        if not np.isnan(cc_index):
            scaled_costs.append(int(source_cost_EU * cc_index))
            countries.append(iso3_code)
    export_df = pd.DataFrame({'iso3_code': countries, 'fixed_cost_scaled': scaled_costs}).set_index('iso3_code', drop=True)
    export_fn = os.path.join('DataDrive', 'ECONOMY', 'PROCESSED')
    if not os.path.exists(export_fn):
        os.makedirs(export_fn)
    export_df.to_csv(os.path.join(export_fn, out))

def scale_dike_maintenance_cost_based_on_construction_index(out = 'scaled_dike_maintenance_cost.csv', source_cost_EU = 0.08E6):
    # load construction price index from Tiggeloven
    path = os.path.join('DataDrive', 'ECONOMY', 'conversion_rates', 'PPP_MER_Con_G109.csv')
    df = pd.read_csv(path, index_col='ISO')
    scaled_costs = []
    countries = []
    for iso3_code, row in df.iterrows():
        cc_index = row['constructioncost_index']
        if not np.isnan(cc_index):
            scaled_costs.append(int(source_cost_EU * cc_index))
            countries.append(iso3_code)
    export_df = pd.DataFrame({'iso3_code': countries, 'fixed_cost_scaled': scaled_costs}).set_index('iso3_code', drop=True)
    export_fn = os.path.join('DataDrive', 'ECONOMY', 'PROCESSED')
    if not os.path.exists(export_fn):
        os.makedirs(export_fn)
    export_df.to_csv(os.path.join(export_fn, out))

if __name__ == '__main__':
    create_scaled_df(out = 'scaled_fixed_migration_cost.csv', source_cost = 250E3, source_country = 'USA')
    create_scaled_df(out = 'scaled_adaptation_cost.csv', source_cost = 10_800, source_country = 'FRA')
    scale_dike_elevation_cost_based_on_construction_index(out = 'scaled_dike_elevation_cost.csv')
    scale_dike_maintenance_cost_based_on_construction_index(out = 'scaled_dike_maintenance_cost.csv')
