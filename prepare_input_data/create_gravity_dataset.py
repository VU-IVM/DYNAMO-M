'''The functions in this script are used to calculate the gravity model in the France case study area.
The source data can be retrieved from: https://github.com/IMAGE-Project/IMAGE_Data/tree/master/France_2006_22
The source of INSEE data is https://www.insee.fr/fr/statistiques/4508111?sommaire=4508161&q=residence 

'''

import pandas as pd
import os
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import pickle


def calculate_scaling_factors_flow():
    # load explanatory variables
    fn_population = os.path.join('DataDrive', 'GRAVITY', 'INSEE', 'Population', 'Departements.csv')
    population_dept = pd.read_csv(fn_population, sep= ';')[['DEP', 'PTOT']].set_index('DEP').astype(str)

    # load population from dataset 
    fn_flows = os.path.join('DataDrive', 'GRAVITY', 'INSEE', 'Flows', 'FD_MIGDEP_2017.csv')
    variablenames =  ['DEPT'] # Only interested in department of birth (DNAI) and current department of residence (DEPT)
    data = pd.read_csv(fn_flows, sep=';')[variablenames].astype(str) 

    # Construct list for filter (data contains string department codes)
    filter_list = [str(i) for i in np.arange(0, 96)] + ['2A', '2B'] # currently Corsica is left out (indicated with 2A and 2B)
    data_filt = data[data['DEPT'].isin(filter_list)]
    data_filt['DEPT'] = [id.zfill(2) for id in data_filt['DEPT']]
    data_filt['DEPT'] = data_filt['DEPT'].replace('2A', '96')
    data_filt['DEPT'] = data_filt['DEPT'].replace('2B', '97')
    

    n_per_dept = data_filt['DEPT'].value_counts()
    pop_data = pd.DataFrame({'dept_id': n_per_dept.index, 'population_in_survey': n_per_dept})
    pop_data = pop_data.set_index('dept_id', drop = True)
    
    # Load coding
    variables_fn = os.path.join('DataDrive', 'GRAVITY', 'INSEE', 'Flows', 'Varmod_MIGDEP_2017.csv')

    variables =  pd.read_csv(variables_fn, sep=';').astype(str)
    variables = variables[variables['COD_VAR'].isin(['DEPT'])][['COD_MOD', 'LIB_MOD']]
    variables = variables[variables['COD_MOD'].isin([id.zfill(2) for id in filter_list])]
    variables['COD_MOD'] = variables['COD_MOD'].replace('2A', '96')
    variables['COD_MOD'] = variables['COD_MOD'].replace('2B', '97')

    variables = variables.set_index('COD_MOD', drop=True)
    
    # Merge names for export
    pop_data['keys'] = variables.loc[[str(admin).zfill(2) for admin in pop_data.index]]
    population = pop_data.set_index('keys', drop = True) 

    # Add fraction of true population for later ure
    flow_scaler = population
    flow_scaler['population_in_department'] = population_dept.loc[flow_scaler.index]['PTOT'].astype(np.int32)
    flow_scaler['frac_of_total'] = flow_scaler['population_in_survey'] / flow_scaler['population_in_department']
    # flow_scaler export pop data for scaling later
    flow_scaler.to_csv(os.path.join('DataDrive', 'GRAVITY', 'INSEE', 'PROCESSED', 'population_in_survey.csv'))

    return flow_scaler

def load_calibration_data_INSEE():
    fn_flows = os.path.join('DataDrive', 'GRAVITY', 'INSEE', 'Flows', 'FD_MIGDEP_2017.csv')
    variablenames =  ['DNAI', 'DEPT', 'IRAN', 'AGEREVQ'] # Only interested in department of birth (DNAI) and current department of residence (DEPT)
    data = pd.read_csv(fn_flows, sep=';')[variablenames].astype(str) # make sure all data is of same type
    original_obs = len(data)
    # Filter to only inlcude internal migration (Foreign and oversee departments are indicated with codes > 95)
    # Construct list for filter (data contains string department codes)
    filter_list = [str(i) for i in np.arange(0, 96)] + ['2A', '2B'] # currently Corsica is left out (indicated with 2A and 2B)
    
    data_filt = data[data['DNAI'].isin(filter_list)]
    data_filt = data_filt[data_filt['DEPT'].isin(filter_list)]
    data_filt['DNAI'] = data_filt['DNAI'].replace('2A', '96')
    data_filt['DNAI'] = data_filt['DNAI'].replace('2B', '97')
    data_filt['DEPT'] = data_filt['DEPT'].replace('2A', '96')
    data_filt['DEPT'] = data_filt['DEPT'].replace('2B', '97')


    data_age_structure  = data_filt[['DEPT', 'AGEREVQ']]

    # Filter based on previous department of residence (only include people that moved between departments in the previous year)
    data_filt = data_filt[data_filt['IRAN'].isin(['4', '5'])]

    # obs_of people that moved
    moved_last_year = len(data_filt)

    # Also filter out moves within own department (although this may be useful too)
    data_filt = data_filt[data_filt['DNAI'] != data_filt['DEPT']]
    print(f'Original sample size: {original_obs}')
    print(f'Sample size after all filters: {len(data_filt)}')
    print(f'Percentage of sample moved in last year: {round(moved_last_year/ original_obs * 100, 3)}%')
    print(f'Percentage of sample moved in to another department: {round(len(data_filt)/ original_obs * 100, 3)}%')

    # Sum DNAI DEPT pairs to derive total annual flow
    data_filt['DNAI_DEPT'] = data_filt['DNAI'] + '_' + data_filt['DEPT']

    # Sum pairs
    flow = data_filt['DNAI_DEPT'].value_counts()

    # Split pairs again and store
    DNAI = [code.split('_', 2)[0].zfill(2) for code in flow.index]
    DEST = [code.split('_', 2)[1].zfill(2) for code in flow.index]


    # Load in metadata to convert department codes to department names
    variables_fn = os.path.join('DataDrive', 'GRAVITY', 'INSEE', 'Flows', 'Varmod_MIGDEP_2017.csv')

    variables =  pd.read_csv(variables_fn, sep=';')
    variables = variables[variables['COD_VAR'].isin(['DEPT'])][['COD_MOD', 'LIB_MOD']]
    variables = variables[variables['COD_MOD'].isin([str(i).zfill(2) for i in np.arange(1, 96)] + ['2A', '2B'])]
    variables['COD_MOD'] = variables['COD_MOD'].replace('2A', '96')
    variables['COD_MOD'] = variables['COD_MOD'].replace('2B', '97')

    variables = variables.set_index(variables['COD_MOD'], drop=True)
    
    # Merge names for export
    admin_origin = list(variables.loc[DNAI]['LIB_MOD'])
    admin_destination = list(variables.loc[DEST]['LIB_MOD'])

    # construct and export
    flows_dept_admin = pd.DataFrame({'origin': admin_origin, 'destination': admin_destination, 'flow': np.array(flow)}).set_index('origin', drop=True)


    # Scale flows to fraction of population answering survey NO LONGER USED
    # flow_scaler = calculate_scaling_factors_flow()


    out_folder = os.path.join('DataDrive', 'GRAVITY', 'INSEE', 'PROCESSED')
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    # Now create empty dataset with dep-dep pairs and fill with flow data. No flow = flow of 0 individual(s)
    # load all departments
    orig_dest_pairs = []
    # extract all unique departments included
    all_departments = np.unique(list(flows_dept_admin.index) + list(flows_dept_admin['destination']))

    for orig in all_departments:
        for dest in all_departments:
            if orig != dest:
                orig_dest_pairs.append(f'{orig}_{dest}')
    
    flow_fill = np.full(len(orig_dest_pairs), 0, dtype=np.int64)
    
    flows_dept_filled = pd.DataFrame({'flow': flow_fill}, index=orig_dest_pairs)
    
    flows_dept_admin_paired = pd.DataFrame(
        {'orig_dest_pairs': list(flows_dept_admin.index + '_' + flows_dept_admin['destination']),
        'flow': list(flows_dept_admin['flow'])})
    
    for _, pair in flows_dept_admin_paired.iterrows():
        flows_dept_filled.loc[pair['orig_dest_pairs']] = pair['flow']
    
    orig = [pair.split('_', 2)[0].zfill(2) for pair in flows_dept_filled.index]
    dest = [pair.split('_', 2)[1].zfill(2) for pair in flows_dept_filled.index]

    flows_dept_export = pd.DataFrame({'origin': orig, 'destination': dest, 'flow': list(flows_dept_filled['flow'])}).set_index('origin', drop=True)

    flows_dept_export.to_csv(os.path.join(out_folder, 'department_migration_flows_2017.csv'))

    # Now process age structure
    data_age_structure = data_age_structure.loc[data_age_structure['AGEREVQ'] != 'ZZ']
    data_age_structure['AGEREVQ'] = np.array(data_age_structure['AGEREVQ'], dtype = np.int32)
    mean_age_department = data_age_structure.groupby('DEPT')['AGEREVQ'].mean()
    indices = [str(i).zfill(2) for i in mean_age_department.index]
    export_age = pd.DataFrame({'key':variables.loc[indices]['LIB_MOD'], 'median_age': list(mean_age_department)})
    export_age.to_csv(os.path.join(out_folder, 'department_mean_age_2017.csv'), index=False)

def produce_gravity_dataset(scaled=False):
    # load flow dataset
    flow_dataset = pd.read_csv(os.path.join('DataDrive', 'GRAVITY', 'INSEE', 'PROCESSED', 'department_migration_flows_2017.csv'))
    origin_keys = np.array(flow_dataset['origin'])
    destination_keys = np.array(flow_dataset['destination'])
    age_dataset = pd.read_csv(os.path.join('DataDrive', 'GRAVITY', 'INSEE', 'PROCESSED', 'department_mean_age_2017.csv'), index_col = 'key')
    # load explanatory variables
    fn_population = os.path.join('DataDrive', 'GRAVITY', 'INSEE', 'PROCESSED', 'population_in_survey.csv')
    population = pd.read_csv(fn_population).set_index('keys', drop=True)

    if scaled:
        population = population['population_in_department_scaled']
    else:
        population = population['population_in_department']

    # Add fraction of true population for later ure
    fn_income = os.path.join('DataDrive', 'GRAVITY', 'INSEE', 'Income', 'base-cc-filosofi-2016.xls')
    income = pd.read_excel(fn_income, sheet_name='DEP', skiprows=5)[['LIBGEO', 'MED16']].set_index('LIBGEO')


    # Create pop frame for both origin and destination
    pop_origin = np.array(population.loc[origin_keys], dtype=np.float32)
    pop_dest = np.array(population.loc[destination_keys], dtype=np.float32)

    # Create income array for both origin and destination
    inc_origin = np.array(income.loc[origin_keys], dtype=np.float32)
    inc_dest = np.array(income.loc[destination_keys], dtype=np.float32)
    
    # Create age arrays for both origin and destination
    age_origin = np.array(age_dataset.loc[origin_keys], dtype=np.float32)
    age_destination = np.array(age_dataset.loc[destination_keys], dtype=np.float32)


    # loading coastline and reprojecting to correct system
    fn_coastline = os.path.join('DataDrive', 'AMENITIES', 'Europe_coastline', 'Europe_coastline.shp')
    coastline_gpd = gpd.read_file(fn_coastline).to_crs(epsg = 4326)

    # loading admin shapefiles for overlaying and distance calculation
    fn_gadm = os.path.join('DataDrive', 'GADM', 'gadm36_2.shp')
    gadm_gpd = gpd.read_file(fn_gadm)
    gadm_gpd = gadm_gpd[gadm_gpd['GID_0'] == 'FRA'][['NAME_2', 'geometry']]
    coastal_admins = np.array(gpd.overlay(gadm_gpd, coastline_gpd, how="intersection", keep_geom_type=False)['NAME_2'])
 
    # Create list of coastal departments
    gadm_gpd['centroid'] = gadm_gpd.to_crs(epsg = 3857).centroid

    # Create array with distances between centroids of departments
    dist = np.full(origin_keys.shape[0], -1, dtype = np.float32)
    coastal_i = np.full(origin_keys.shape[0], 0, dtype = np.float32)
    coastal_j = np.full(origin_keys.shape[0], 0, dtype = np.float32)

    i = 0
    for origin, destination in zip(origin_keys, destination_keys):
        # Subset
        origin_loc = gadm_gpd[gadm_gpd['NAME_2'] == origin]['centroid']
        destination_loc = gadm_gpd[gadm_gpd['NAME_2'] == destination]['centroid']
        
        # Create shapely point objects
        origin_point = Point(origin_loc.x._values[0],  origin_loc.y._values[0])
        destination_point = Point(destination_loc.x._values[0],  destination_loc.y._values[0])

        # Calculate distance[km] and store
        dist[i] = origin_point.distance(destination_point)/1E3

        # Now also check if intersecting with coastline
        if origin in coastal_admins:
            coastal_i[i] = 1
        if destination in coastal_admins:
            coastal_j[i] = 1

        i += 1

    explanatory_variables = {
    'population_i': pop_origin.flatten(),
    'population_j':pop_dest.flatten(),
    'age_i': age_origin.flatten(),
    'age_j': age_destination.flatten(),
    'income_i': inc_origin.flatten(),
    'income_j': inc_dest.flatten(),
    'coastal_i': coastal_i.flatten(),
    'coastal_j': coastal_j.flatten(),
    'distance': dist.flatten()
    } 

    # Save 
    outfile = os.path.join('DataDrive', 'GRAVITY', 'INSEE', 'PROCESSED', 'variables_FRA.pickle')
    f = open(outfile,"wb")
    pickle.dump(explanatory_variables, f)
    f.close()

if __name__ == '__main__':
    load_calibration_data_INSEE()
    produce_gravity_dataset()
