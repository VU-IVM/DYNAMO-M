import pandas as pd
import os
import geopandas as gpd
import sys
sys.path.append('..')
import pickle
import numpy as np
import yaml

def read_data(fp = os.path.join('DataDrive', 'SLR', 'copy_of_global_flood_depth-damage_functions__30102017.xlsx')):
 
    agents_fn = r'/scistor/ivm/ltf200/COASTMOVE/DataDrive/SLR/households_gdl_2015/xxlarge'
    countries = np.unique([region[:3] for region in os.listdir(agents_fn)])

    # load model regions
    regions = yaml.load(open(r'/scistor/ivm/ltf200/COASTMOVE/utils/model_regions_original.yml'), Loader=yaml.FullLoader)        

    # filter on countries in model
    max_damages = pd.read_excel(fp, sheet_name='MaxDamage-Residential', skiprows=1).dropna().set_index('Country', drop = True)
    max_damages_industrial = pd.read_excel(fp, sheet_name='MaxDamage-Industrial', skiprows=1).dropna().set_index('Country', drop = True)
    max_damages_commercial = pd.read_excel(fp, sheet_name='MaxDamage-Commercial', skiprows=1).dropna().set_index('Country', drop = True)
    max_damages_residential = pd.read_excel(fp, sheet_name='MaxDamage-Residential', skiprows=1).dropna().set_index('Country', drop = True)


    max_damages.index = [country.upper() for country in max_damages.index]
    max_damages_industrial.index = [country.upper() for country in max_damages_industrial.index]
    max_damages_commercial.index = [country.upper() for country in max_damages_commercial.index]
    max_damages_residential.index = [country.upper() for country in max_damages_residential.index]
    iso_3_conversion = pd.read_excel(fp, sheet_name='ISO_Table').set_index('A 3', drop = True)

    # get CPI for converting to 2015 values
    CPI_fn = os.path.join('DataDrive', 'ECONOMY', 'conversion_rates', 'CPI.csv')
    CPI_df = pd.read_csv(CPI_fn, skiprows=4, index_col='Country Code')[['2015']]
    
    # preallocate indice an max dam lists 
    iso3 = []
    region = []
    max_damage_object = []
    max_damage_residential = []
    max_damage_industrial = []
    max_damage_commercial = []

    country_names = []
    for country in countries:
        # translate
        if country not in iso_3_conversion.index:
            continue
        
        country_name = iso_3_conversion.loc[country]['Country'].upper()
        if country_name.upper() in max_damages.index:
            # get data and store
            iso3.append(country)
            country_names.append(country_name)
            # get conversion to 2015 EUR
            conversion_rate = CPI_df.loc[country].iloc[0]/ 100
            if np.isnan(conversion_rate):
                conversion_rate = 1
            max_damage_object.append(max_damages.loc[country_name]['Total.2'] * conversion_rate)
            max_damage_industrial.append(max_damages_industrial.loc[country_name]['Total.1'] * conversion_rate)
            max_damage_commercial.append(max_damages_commercial.loc[country_name]['Total.1'] * conversion_rate)
            max_damage_residential.append(max_damages_residential.loc[country_name]['Total.1'] * conversion_rate)

            # get region correpsonding to country
            success = False
            for key in regions:
                if country in regions[key]:
                    region.append(key)
                    success=True
                    break
            if not success:
                raise ValueError(f'{country} not in regions')
        else:
            print(f'{country} not in max damage')
    # construct dataframe

    max_damage_values = pd.DataFrame({'Country': country_names, 'iso3_code': iso3, 'region': region, 'max_damage_residential_object': max_damage_object, 'max_damage_residential_LU': max_damage_residential, 'max_damage_industrial_LU': max_damage_industrial, 'max_damage_commercial_LU': max_damage_commercial})
    target_folder = os.path.join('DataDrive', 'damage_functions', 'PROCESSED') 
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    max_damage_values.to_csv(os.path.join(target_folder, 'max_damage_countries.csv'), index=False)

if __name__ == '__main__':
    read_data()