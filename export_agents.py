import numpy as np
import geopandas as gpd
from numpy.core.fromnumeric import size
import pandas as pd
from honeybees.library.raster import write_to_array 
import os
from osgeo import gdal, osr 
import fiona
import rasterio
import rasterio.mask



def export_agents(agent_locs, agent_size, agent_adapt, agent_income, agent_risk_perception, agent_ead,
            agent_since_flood, year, export=False, report_folder=None):
    
    '''This function exports the agents and certain agent attributes to a csv for further visualization '''

    # Initiate array for appending ##REPLACE APPEND WITH CONCAT, depreciation warning
    all_agents = pd.DataFrame()
    hh_index = []
    j = 0
    # filter out inland nodes
    for i in agent_locs:
        if len(i) > 2:
            all_agents = all_agents.append(pd.DataFrame(i))
            hh_index.append(j)
        j += 1
    
    hh_sizes = [agent_size[i] for i in hh_index]
    hh_adapt = [agent_adapt[i] for i in hh_index]
    hh_income = [agent_income[i] for i in hh_index]
    hh_risk_perception = [agent_risk_perception[i] for i in hh_index]
    hh_ead = [agent_ead[i] for i in hh_index]
    hh_since_flood =  [agent_since_flood[i] for i in hh_index]

    size_array = np.array([])
    adapt_array = np.array([])
    income_array = np.array([])
    risk_perception_array = np.array([])
    ead_array = np.array([])
    since_flood_array = np.array([])


    for i in range(len(hh_index)):
        size_array = np.append(size_array, np.array(hh_sizes[i]))
        adapt_array = np.append(adapt_array, np.array(hh_adapt[i]))
        income_array = np.append(income_array, np.array(hh_income[i]))
        risk_perception_array = np.append(risk_perception_array, np.array(hh_risk_perception[i]))
        ead_array = np.append(ead_array, np.array(hh_ead[i]))
        since_flood_array = np.append(since_flood_array, np.array(hh_since_flood[i]))

    all_agents['Size'] = size_array
    all_agents['Adapted'] = adapt_array
    all_agents['Income'] = income_array
    all_agents['RiskPerception'] = risk_perception_array
    all_agents['EAD'] = ead_array
    all_agents['YrsSinceFlood'] = since_flood_array
    all_agents['Index'] = all_agents.index
    all_agents = all_agents.set_index('Index')
    
    # export to csv for testing
    if export == True:
        destination = os.path.join(report_folder, 'all_agents')
        if not os.path.exists(destination): 
            os.mkdir(destination)
        all_agents.to_csv(os.path.join(destination, f'agent_out_{year}.csv')) # --> need to translate this to raster with population density


def export_matrix(geoms, dest_folder, move_dictionary, year):
    '''This function exports the simulated migration matrix to csv for further processing in GIS or other. It takes the geoms, destination folder, move dictionary and current year as input arguments.'''
    
    # First extract all GADM names to list
    
    names =[geoms[i]['properties']['id'] for i in range(len(geoms))]

    # Export merged move dictonary to csv (annual migration matrix)
    export_matrix = pd.DataFrame(move_dictionary)
    export_matrix = export_matrix[['from', 'to']]
    # We are only interested in numbers of migrants
    export_matrix['combined'] = export_matrix['from'].astype(str) + '_' + export_matrix['to'].astype(str)
    pairs, flows = np.unique(np.array(export_matrix['combined']), return_counts = True)
    from_region = [int(comb.split('_')[0]) for comb in pairs]
    to_region = [int(comb.split('_')[1]) for comb in pairs]
    from_region_name = [geoms[i]['properties']['id'] for i in from_region]
    to_region_name = [geoms[i]['properties']['id'] for i in to_region]
    matrix = pd.DataFrame()
    matrix['from'] = from_region_name
    matrix['to'] = to_region_name
    matrix['flow'] = flows
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    matrix.set_index('from').to_csv(os.path.join(dest_folder, f'migration_matrix_{year}.csv'))

def export_agent_array(household_size, household_coords, year):
    '''This function writes the agent population to an array, creating a population density map'''
    # Create array to write to
    # Fill cropped population density array (to maintain geo metadata)
    target_tiff =  r'DataDrive/POPULATION/SimPopDens.tif'

    if not os.path.exists(target_tiff):
        ras_fp = r'DataDrive/POPULATION/GHS_POP_2015.tif'
        shp_fp = r'DataDrive/SLR/GADM/GADM_1.shp'
    
        
        with fiona.open(shp_fp, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]

        
        with rasterio.open(ras_fp) as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True) # Masking using FRANCE
            out_meta = src.meta
        
        out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})


        with rasterio.open(target_tiff, "w", **out_meta) as dest:
            dest.write(out_image)

    
    household_size = household_size[96:] # hard coded to France (quick fix, should find way to filter nested list). We only want coastal regions here
    household_coords = household_coords[96:]

    # Extract to numpy arrays
    household_size_arr = np.array([])
    for i in household_size:
        household_size_arr = np.append(household_size_arr, i)

    household_coords_arr = np.array([])
    for i in household_coords:
        household_coords_arr = np.append(household_coords_arr, i)
    
    household_coords_arr = household_coords_arr.reshape(household_coords_arr.size//2, 2)


    src = gdal.Open(target_tiff)
    transform = src.GetGeoTransform()
    myarray = np.array(src.GetRasterBand(1).ReadAsArray())

    array = np.zeros(myarray.shape)
    write_to_array(array = array, values = household_size_arr, coords = household_coords_arr, gt = transform)
    
    with rasterio.open(target_tiff) as src:
        out_meta = src.meta

    out_meta['nodata'] = 0

    target_tiff =  f'DataDrive/POPULATION/SimPopDens_{year}.tif'
    with rasterio.open(target_tiff, "w", **out_meta) as dest:
        dest.write(array, indexes=1) 

def export_pop_change(start, end):
    start_tiff = f'DataDrive/POPULATION/SimPopDens_{start}.tif'
    end_tiff = f'DataDrive/POPULATION/SimPopDens_{end}.tif'

    src = gdal.Open(start_tiff)
    start_array = np.array(src.GetRasterBand(1).ReadAsArray())
    src = gdal.Open(end_tiff)
    end_array = np.array(src.GetRasterBand(1).ReadAsArray())

    change_array = end_array - start_array

    with rasterio.open(start_tiff) as src:
        out_meta = src.meta
     
    target_tiff =  f'DataDrive/POPULATION/SimPopDens_change.tif'
    with rasterio.open(target_tiff, "w", **out_meta) as dest:
        dest.write(change_array, indexes=1) 

    # Export

