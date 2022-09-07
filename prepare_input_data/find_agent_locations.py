import geopandas as gpd
from osgeo import gdal, ogr
# from hyve.library.raster import rasterize
import rasterio
import rasterio.mask
import rasterio.transform
import os
from numba import njit
import subprocess
import numpy as np
from shapely.ops import cascaded_union
from shapely.ops import unary_union
from osgeo import osr

from random import randint

def rasterize(input_fn, output_fn, xsize, ysize, projection, gt, attribute, nodatavalue, dtype):
    shp = ogr.Open(input_fn)
    shp_lyr = shp.GetLayer()

    output = gdal.GetDriverByName('GTiff').Create(
        output_fn,
        xsize,
        ysize,
        1,
        dtype,
        options=['COMPRESS=NONE'] # number of GADM-2 areas exceeds maximum possible under int16 
    )

    if isinstance(projection, int):
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(projection)
        projection = srs.ExportToWkt()

    output.SetProjection(projection)
    output.SetGeoTransform(gt)

    band = output.GetRasterBand(1)
    band.SetNoDataValue(nodatavalue)
    gdal.RasterizeLayer(output, [1], shp_lyr, options=[f"ATTRIBUTE={attribute}"])

    band = None
    output = None
    shp = None





LEVEL = 2
YEAR = 2015

@njit
def generate_locations(population, gadm, can_flood, x_offset, y_offset, x_step, y_step, max_household_size):

    household_locations = np.full((population.sum(), 2), -1, dtype=np.float32)
    household_sizes = np.full(population.sum(), -1, dtype=np.int32)
    household_admin = np.full(population.sum(), -1, dtype=np.int32)

    total_household_count = 0
    for row in range(0, population.shape[0]):
        for col in range(0, population.shape[1]):

            cell_population = population[row, col]
            if cell_population != 0 and can_flood[row, col]:

                ymax = y_offset + row * y_step
                ymin = ymax + y_step

                xmin = x_offset + col * x_step
                xmax = xmin + x_step

                n_households = 0
                while cell_population > 0:
                    household_size = randint(1, max_household_size)  # create
                    household_size = min(household_size, cell_population)  # cap household size to current population left in cell
                    household_sizes[total_household_count+n_households] = household_size
                    cell_population -= household_size
                    n_households += 1
                
                
                household_locations[total_household_count:total_household_count+n_households, 0] = np.random.uniform(xmin, xmax, size=n_households)
                household_locations[total_household_count:total_household_count+n_households, 1] = np.random.uniform(ymin, ymax, size=n_households)
                household_admin[total_household_count:total_household_count+n_households] = gadm[row, col]

                total_household_count += n_households

    return household_locations[:total_household_count], household_sizes[:total_household_count], household_admin[:total_household_count]


@njit
def create_people(max_household_size, household_size_region):
    indices = np.full((household_size_region.size, max_household_size), -1, dtype=np.int32)
    people_index = 0
    for i in range(household_size_region.size):
        household_size = household_size_region[i]
        for j in range(household_size):
            indices[i, j] = people_index
            people_index += 1
    gender = np.random.randint(0, 2, size=people_index)
    
    age = np.random.normal(30, 20, size=people_index)
    age[age<0] = 0
    age = age.astype(np.int8)
    
    
    return indices, gender, age


def main(max_household_size):
    # if location == 'bangladesh':
    #     countries_iso3 = ["BGD"]
    # elif location == 'se-asia':
    #     countries_iso3 = ["BGD", 'THA', 'MMR']
    # elif location == 'global':
    #     countries_iso3 = None
    # else:
    #     raise NotImplementedError

    agent_folder = f'DataDrive\SLR\households_gadm_{LEVEL}_{YEAR}'
    try:
        os.makedirs(agent_folder)
    except OSError:
        pass
        
    # area_fn = f'DataDrive/SLR/admin/area_{location}_{LEVEL}.shp'
    # if not os.path.exists(area_fn):
    #     gadm_path = f'DataDrive/SLR/GADM/GADM_{LEVEL}.shp'
    #     area = gpd.GeoDataFrame.from_file(gadm_path)
    #     if countries_iso3:
    #         area = area[area['GID_0'].isin(countries_iso3)].reset_index(drop=True)
    #         boundary = gpd.GeoDataFrame([[f'{location}', area.geometry.unary_union]], columns=['id', 'geometry'], crs=area.crs)
    #         boundary.to_file(f'DataDrive/SLR/admin/{location}_boundary.shp')
        
    #     can_flood_shp = gpd.GeoDataFrame.from_file('DataDrive/SLR/admin/can_flood_gadm_1_merged.shp')
        
    #     area.to_file(area_fn)

    population_path = f'DataDrive/POPULATION/GHS_POP_{YEAR}.tif'
    with rasterio.open(population_path) as src:
        population = src.read(1).astype(np.int16)
        transform = src.transform
        
    with rasterio.open(f'DataDrive/SLR/GADM/GADM_{LEVEL}.tif') as gadm_src:
        gadm_profile = gadm_src.profile
        gadm = gadm_src.read(1)
        transform = gadm_src.transform
    
    population[gadm == -1] = 0

    with rasterio.open('DataDrive/SLR/can_flood.tif') as src:
        can_flood = src.read(1)
        can_flood[(np.isnan(can_flood)) | (can_flood < 0.001)] = 0
        can_flood = can_flood > 0

    gadm_max = np.max(gadm)
    can_flood_gadm = gadm.copy()
    can_flood_gadm[can_flood == True] += gadm_max
    with rasterio.open(f'DataDrive/SLR/can_flood_gadm_{LEVEL}.tif', 'w', **gadm_profile) as dst:
        dst.write(can_flood_gadm, 1)

    population = population.astype(np.int64)
    population[population < 0] = 0
    population[can_flood == False] = 0

    gt = transform.to_gdal()

    x_offset = gt[0]
    y_offset = gt[3]
    x_step = gt[1]
    y_step = gt[5]
    
    household_locations, household_size, household_admin = generate_locations(population, gadm, can_flood, x_offset, y_offset, x_step, y_step, max_household_size)

    gdf = gpd.GeoDataFrame.from_file(f'DataDrive/SLR/GADM/GADM_{LEVEL}.shp')
    ID = gdf[f'GID_{LEVEL}'].to_numpy().astype("str")

    idx = np.argsort(household_admin)
    household_admin = household_admin[idx]
    household_admin = np.take(ID, household_admin)
    household_locations = household_locations[idx]
    household_size = household_size[idx]

    keys, start_indices = np.unique(household_admin, return_index=True)
    idx = np.argsort(start_indices)
    start_indices = start_indices[idx]
    
    keys = keys[idx]
    start_indices = np.append(start_indices, np.array(household_admin.size))
    for i in range(start_indices.size - 1):
        key = keys[i]
        start_idx, end_idx = start_indices[i], start_indices[i+1]

        for size, step in [('small', 1_000_000), ('medium', 1_000), ('large', 100), ('xlarge', 10), ('xxlarge', 1)]:
            household_locations_region = household_locations[start_idx:end_idx:step]
            household_size_region = household_size[start_idx:end_idx:step]
            people_indices, gender, age = create_people(max_household_size, household_size_region)

            subfolder = os.path.join(agent_folder, size, key)
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)

            np.save(os.path.join(subfolder, 'locations.npy'), household_locations_region)
            np.save(os.path.join(subfolder, 'size.npy'), household_size_region)
            np.save(os.path.join(subfolder, 'people_indices.npy'), people_indices)
            np.save(os.path.join(subfolder, 'gender.npy'), gender)
            np.save(os.path.join(subfolder, 'age.npy'), age)

def create_admin_tifs():
    try:
        os.makedirs('DataDrive/SLR/GADM')
    except OSError:
        pass
    for LEVEL in [1, 2]:
        print(LEVEL)
        output_shp = f'DataDrive/SLR/GADM/GADM_{LEVEL}.shp'
        output_tif = f'DataDrive/SLR/GADM/GADM_{LEVEL}.tif'
        population_fn = f'DataDrive/POPULATION/GHS_POP_{YEAR}.tif'
        if not os.path.exists(output_tif):
            if not os.path.exists(output_shp):
                gdf = gpd.GeoDataFrame.from_file(f'DataDrive/GADM/gadm36_{LEVEL}.shp')
                gdf = gdf[gdf['GID_0'] == 'FRA'] # Only select France
                gdf = gdf.reset_index()
                gdf['ID'] = gdf.index
                gdf.to_file(output_shp)
            with rasterio.open(population_fn) as src:
                rasterize(
                    output_shp,
                    output_tif,
                    src.width,
                    src.height,
                    4326,
                    src.transform.to_gdal(),
                    'id',
                    -1,
                    gdal.GDT_Int32
                )


def create_shapefiles():
    tif_file = f'DataDrive/SLR/admin/can_flood_gadm_{LEVEL}.tif'
    if not os.path.exists(tif_file):
        with rasterio.open(f'DataDrive/SLR/GADM/GADM_{LEVEL}.tif') as gadm_src:
            gadm_profile = gadm_src.profile
            gadm = gadm_src.read(1)
            gadm_max = np.max(gadm)
        
        with rasterio.open('DataDrive/SLR/can_flood.tif') as src:
            can_flood = src.read(1)
            can_flood[(np.isnan(can_flood)) | (can_flood < 0.001)] = 0
            can_flood[gadm == gadm_profile['nodata']] = 0
            can_flood = can_flood > 0

        can_flood_gadm = gadm.copy()
        can_flood_gadm[can_flood == True] += gadm_max
        with rasterio.open(tif_file, 'w', **gadm_profile) as dst:
            dst.write(can_flood_gadm, 1)

    shp_file = tif_file.replace('.tif', '.shp')
    if not os.path.exists(shp_file):
        subprocess.call(
            r"python  C:\Users\ltf200\Anaconda3\envs\abm\Scripts\gdal_polygonize.py" + f" {tif_file} {shp_file}",
            shell=True
        )

    IDs = gpd.GeoDataFrame.from_file(f'DataDrive/SLR/GADM/GADM_{LEVEL}.shp')[f'GID_{LEVEL}'].to_numpy().astype(str)
    n_IDs = IDs.size - 1
    
    gdf = gpd.GeoDataFrame.from_file(shp_file)
    ids, geoms, keys = [], [], []
    grouped_by_DN = gdf.groupby(by='DN')
    for DN, geometries in grouped_by_DN:
        if any(geometries.is_valid) == False:
            geometries = geometries.buffer(0)
            geoms.append(unary_union(geometries.geometry))
            geoms.append(unary_union(geometries['geometry']))
       
        # geometries = geometries.buffer(0)
        # geoms.append(unary_union(geometries.geometry))

        ids.append(DN)
        if DN > n_IDs:
            keys.append(f"{IDs[DN-n_IDs]}_flood_plain")
        else:
            keys.append(IDs[DN])

        
       
    
    gdf_out = gpd.GeoDataFrame({'ID': ids, 'keys': keys, 'geometry': geoms}, crs="EPSG:4326")
    gdf_out.to_file(shp_file.replace('.shp', '_merged.shp'))

if __name__ == '__main__':
    create_admin_tifs()
    # create_shapefiles()
    # main(max_household_size=10)
    # main('bangladesh')
    # main('global')
    