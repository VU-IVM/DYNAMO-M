import pandas as pd
import numpy as np
import os
import geopandas as gpd
import matplotlib.pyplot as plt


####################################################
# transform current gamd to name used in synth pop #
####################################################

# load geom files
geom_id = 'ESP.1_1_flood_plain'
GDL_shape = gpd.read_file(os.path.join('DataDrive', 'AGENTS', 'synthPOP', 'shapes', 'GDL Shapefiles V6.2 large.shp'))
GADM_shape = gpd.read_file(os.path.join('DataDrive', 'SLR', 'GADM', 'GADM_1.shp'))

# filter on isocode and kee relative columns
GDL_shape = GDL_shape[GDL_shape['iso_code'].apply(lambda x: x[:3] in ['ESP'])][['gdlcode', 'geometry']]
GADM_shape = GADM_shape[GADM_shape['GID_0'].apply(lambda x: x[:3] in ['ESP'])][['GID_1', 'geometry']]

# get centroids
GADM_shape.geometry = GADM_shape.geometry.centroid
# join datasets 
shapes_joined = gpd.sjoin(GDL_shape, GADM_shape)[['gdlcode', 'GID_1']]

# load translater
translator_fn = os.path.join('DataDrive', 'AGENTS', 'synthPOP', 'data', 'GDL_match_population_all_LIS.csv')
translator = pd.read_csv(translator_fn, sep = ';')

# translate current geom id to file to load
geom_id = 'ESP.12_1'

# first use geom code to get gdl code
gdl_region = shapes_joined[shapes_joined['GID_1'] == geom_id]['gdlcode'].values[0]

# get reg number
region_nr = int(translator[translator['GDLcode'] == gdl_region]['GEOLEV1_region_number'].values[0])



#########################
#### read data files ####
#########################

region = 'it14'
n_columns = 11

data_fn = os.path.join('DataDrive', 'AGENTS', 'synthPOP', 'data', f'{region}_may23_synthpop_reg1.dat')

# load in np
data_np = np.fromfile(data_fn, dtype=np.int32)

# get individuals
n_people = data_np.size// 11

# reshapa data
data_reshaped = np.reshape(data_np, (n_columns, n_people)).transpose()

# construct pd for testing
colnames = ['INCOME', 'RURAL', 'FARMING', 'AGECAT', 'GENDER', 
                         'EDUCAT', 'HHTYPE', 'HHID', 'RELATE', 'HHSIZECAT', 'GEOLEV1']

data_pd = pd.DataFrame(data_reshaped, columns=colnames)
