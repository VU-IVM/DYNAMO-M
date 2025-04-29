import geopandas as gpd
from osgeo import gdal, ogr
import rasterio
import rasterio.mask
import rasterio.transform
import os
from numba import njit
import subprocess
import numpy as np
from shapely.ops import unary_union
from osgeo import osr
import geopandas as gpd
from random import randint
from prepare_synthetic_agents import read_synthetic_population_data

YEAR = 2015
# adjust to user
GDAL_POLYGONIZE = r"C:\Users\ltf200\AppData\Local\miniconda3\envs\abm\Scripts\gdal_polygonize.py"
# assert os.path.exists(GDAL_POLYGONIZE) # results in failure to build docs

def rasterize_burn(
        input_fn,
        output_fn,
        xsize,
        ysize,
        projection,
        gt,
        burn_value,
        nodatavalue,
        dtype,):
    
    shp = ogr.Open(input_fn)
    shp_lyr = shp.GetLayer()

    output = gdal.GetDriverByName('GTiff').Create(
        output_fn,
        xsize,
        ysize,
        1,
        dtype,
        # number of GADM-2 areas exceeds maximum possible under int16
        options=['COMPRESS=LZW']
    )

    if isinstance(projection, int):
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(projection)
        projection = srs.ExportToWkt()

    output.SetProjection(projection)
    output.SetGeoTransform(gt)

    band = output.GetRasterBand(1)
    band.SetNoDataValue(nodatavalue)
    gdal.RasterizeLayer(output, [1], shp_lyr, options=[
                        f"burnValues ={burn_value}"])

    band = None
    output = None
    shp = None
     

def rasterize(
        input_fn,
        output_fn,
        xsize,
        ysize,
        projection,
        gt,
        attribute,
        nodatavalue,
        dtype):
    shp = ogr.Open(input_fn)
    shp_lyr = shp.GetLayer()

    output = gdal.GetDriverByName('GTiff').Create(
        output_fn,
        xsize,
        ysize,
        1,
        dtype,
        # number of GADM-2 areas exceeds maximum possible under int16
        options=['COMPRESS=LZW']
    )

    if isinstance(projection, int):
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(projection)
        projection = srs.ExportToWkt()

    output.SetProjection(projection)
    output.SetGeoTransform(gt)

    band = output.GetRasterBand(1)
    band.SetNoDataValue(nodatavalue)
    gdal.RasterizeLayer(output, [1], shp_lyr, options=[
                        f"ATTRIBUTE={attribute}"])

    band = None
    output = None
    shp = None


@njit
def generate_locations(
    total_household_count,
    household_locations,
    household_sizes,
    household_types,
    household_incomes,
    household_admin,
    gdl_idx,
    synth_household_sizes,
    synth_household_types,
    synth_income_perc,
    population: np.ndarray,
    gdl: np.ndarray,
    can_flood: np.ndarray,
    x_offset: float,
    y_offset: float,
    x_step: float,
    y_step: float,
    max_household_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''This function is used to create the locations of the household agent and household sizes. It generates households by sampling from the gridded population map using random household sizes.

    Args:
        population: array constructed from gridded population data.
        can_flood: array representing a global raster. It is a boolean contructed using the 1/100 year flood map of 2080. 1 = inundated, 0 = not inundated.
        x_offset: geotransformation
        y_offset: geotransformation
        x_step: x-dimension of cell size in degrees
        y_step: y-dimension of cell size in degrees
        max_household_size: the maximimum household size used in sampling household agents.

    Returns:
        household_locations: array containing coordinates of each generated household.
        household_sizes: array containing the household size of each generated household.
        household_admin: array containing the value of the rasterized admin unit the household resides in.

    '''

    # only iterate over cells in floodplain and that lie within gdl geom
    flood_plain_cells = np.where(np.logical_and(can_flood == True, gdl == gdl_idx))
    hh_types_random = np.array([1, 2, 3, 3, 3]) # 1: single adult, 2: two adults, 3: two adults + children
    hh_sizes_random = np.array([1, 2, 3, 6, 7])

    for row, col in zip(flood_plain_cells[0], flood_plain_cells[1]):
        assert can_flood[row, col]
        assert gdl[row, col] == gdl_idx

        cell_population = population[row, col]
        if cell_population != 0 and can_flood[row, col]:

            ymax = y_offset + row * y_step
            ymin = ymax + y_step

            xmin = x_offset + col * x_step
            xmax = xmin + x_step

            n_households = 0
            while cell_population > 0:
                # if not inlcuded in synthpop create random size and income
                if synth_household_sizes.size == 0:
                    household_indice_no_data = np.random.randint(hh_types_random.size) 
                    household_size = hh_sizes_random[household_indice_no_data]  # create (set household size for regions not in data to 6 for now)
                    household_type = hh_types_random[household_indice_no_data]
                    household_income_perc = np.random.randint(100)
                # else get household size from synthetic population
                else:
                    # get indice
                    household_indice = np.random.randint(synth_household_sizes.size)
                    #assign
                    household_size = synth_household_sizes[household_indice]
                    household_type = synth_household_types[household_indice]
                    household_income_perc = synth_income_perc[household_indice]
                    # remove
                    synth_household_sizes[household_indice] = synth_household_sizes[-1]
                    synth_household_sizes = synth_household_sizes[:-1]

                    synth_household_types[household_indice] = synth_household_types[-1]
                    synth_household_types = synth_household_types[:-1]

                    synth_income_perc[household_indice] = synth_income_perc[-1]
                    synth_income_perc = synth_income_perc[:-1]

                # cap household size to current population left in cell
                household_size = min(household_size, cell_population)
                household_sizes[total_household_count +
                                n_households] = household_size
                
                # store type
                household_types[total_household_count +
                                n_households] = household_type

                # store income
                household_incomes[total_household_count +
                                n_households] = household_income_perc
               
                cell_population -= household_size
                n_households += 1


            household_locations[total_household_count:total_household_count +
                                n_households, 0] = np.random.uniform(xmin, xmax, size=n_households)
            household_locations[total_household_count:total_household_count +
                                n_households, 1] = np.random.uniform(ymin, ymax, size=n_households)
            household_admin[total_household_count:total_household_count +
                            n_households] = gdl_idx

            total_household_count += n_households

    return total_household_count


@njit
def create_people(
    max_household_size: int,
    household_size_region: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''This function is used to generate the people that populate the households. It also generates certain individual characteristics (gender and age).

    Args:
        max_household_size: the maximum household size.
        household_size_region: an array containing the household sizes of households in the region (coastal node)

    Returns:
        indices: person indices. These indices track the household to which person is a part of (see coastal_node docs).
        gender: array containing the gender of each person (1 or 0).
        age: array containing the age of each person.
    '''

    indices = np.full(
        (household_size_region.size, max_household_size), -1, dtype=np.int32)
    people_index = 0
    for i in range(household_size_region.size):
        household_size = household_size_region[i]
        for j in range(household_size):
            indices[i, j] = people_index
            people_index += 1
    gender = np.random.randint(0, 2, size=people_index)

    age = np.random.normal(30, 20, size=people_index)
    age[age < 0] = 0
    age = age.astype(np.int8)

    return indices, gender, age


def main(max_household_size):
    '''This function samples household agents from the population map. It samples individuals belonging to a household with a random size. '''
    agent_folder = r'DataDrive/SLR/households_gdl_2015'
    try:
        os.makedirs(agent_folder)
    except OSError:
        pass

    population_path = f'DataDrive/POPULATION/GHS_POP_2015.tif'
    with rasterio.open(population_path) as src:
        population = src.read(1).astype(np.int16)
        transform = src.transform

    with rasterio.open(f'DataDrive/SLR/GDL/GDL.tif') as gdl_src:
        gdl_profile = gdl_src.profile
        gdl = gdl_src.read(1)
        gdl_transform = gdl_src.transform
 
    population[gdl == -1] = 0

    with rasterio.open('DataDrive/SLR/can_flood.tif') as src:
        can_flood = src.read(1)
        can_flood[(np.isnan(can_flood)) | (can_flood < 0.001)] = 0
        can_flood = can_flood > 0

    gdl_max = np.max(gdl)
    can_flood_gdl = gdl.copy()
    can_flood_gdl[can_flood] += gdl_max
    with rasterio.open(f'DataDrive/SLR/can_flood_gdl.tif', 'w', **gdl_profile) as dst:
        dst.write(can_flood_gdl, 1)

    population = population.astype(np.int64)
    population[population < 0] = 0
    population[can_flood == False] = 0

    gt = transform.to_gdal()

    x_offset = gt[0]
    y_offset = gt[3]
    x_step = gt[1]
    y_step = gt[5]

    household_locations = np.full((population.sum(), 2), -1, dtype=np.float32)
    household_sizes = np.full(population.sum(), -1, dtype=np.int32)
    household_incomes = np.full(population.sum(), -1, dtype=np.int32)
    household_types = np.full(population.sum(), -1, dtype=np.int32)
    household_admin = np.full(population.sum(), -1, dtype=np.int32)
    total_household_count = 0

    # iterate over gdl_regions 
    for gdl_idx in range(gdl.max() +1):
        data_folder = os.path.join('DataDrive', 'AGENTS', 'PROCESSED', f'r{str(gdl_idx).zfill(4)}')
        if os.path.exists(data_folder):
            synth_household_sizes = np.load(os.path.join(data_folder, 'sizes_region.npy'))
            synth_income_perc = np.load(os.path.join(data_folder, 'income_perc_region.npy'))
            synth_household_types = np.load(os.path.join(data_folder, 'household_type_region.npy'))
        else:
            synth_household_sizes = np.array([], dtype=np.int32)
            synth_income_perc = np.array([], dtype=np.int32)
            synth_household_types = np.array([], dtype=np.int32)

        
        print(f'generating households for region {gdl_idx} of {gdl.max() +1}')
        # Now generate household locations for gdl_idx
        total_household_count = generate_locations(
            total_household_count=total_household_count,
            household_locations=household_locations,
            household_sizes=household_sizes,
            household_types=household_types,
            household_incomes=household_incomes,
            household_admin=household_admin,
            gdl_idx=gdl_idx,
            synth_household_sizes = synth_household_sizes,
            synth_household_types = synth_household_types,
            synth_income_perc = synth_income_perc,
            population=population,
            gdl=gdl,
            can_flood=can_flood,
            x_offset=x_offset,
            y_offset=y_offset,
            x_step=x_step,
            y_step=y_step,
            max_household_size=max_household_size
        )

    # clip redundancy
    household_locations = household_locations[:total_household_count]
    household_sizes = household_sizes[:total_household_count]
    household_types = household_types[:total_household_count]
    household_admin = household_admin[:total_household_count]
    household_incomes= household_incomes[:total_household_count]

    print(f'{household_locations.shape[0]} households created')

    gdf = gpd.GeoDataFrame.from_file(f'DataDrive/SLR/GDL/GDL.shp')
    ID = gdf[f'gdlcode'].to_numpy().astype("str")

    idx = np.argsort(household_admin)
    household_admin = household_admin[idx]
    household_admin = np.take(ID, household_admin)
    household_locations = household_locations[idx]
    household_incomes = household_incomes[idx]
    household_size = household_sizes[idx]
    household_types = household_types[idx]

    keys, start_indices = np.unique(household_admin, return_index=True)
    idx = np.argsort(start_indices)
    start_indices = start_indices[idx]

    keys = keys[idx]
    start_indices = np.append(start_indices, np.array(household_admin.size))
    for i in range(start_indices.size - 1):
        key = keys[i]
        start_idx, end_idx = start_indices[i], start_indices[i + 1]

        for size, step in [('small', 1_000_000), ('medium', 1_000),
                           ('large', 100), ('xlarge', 10), ('xxlarge', 1)]:
            household_locations_region = household_locations[start_idx:end_idx:step]
            household_size_region = household_size[start_idx:end_idx:step]
            household_types_region = household_types[start_idx:end_idx:step]
            household_incomes_region = household_incomes[start_idx:end_idx:step]

            people_indices, gender, age = create_people(
                max_household_size, household_size_region)
            subfolder = os.path.join(agent_folder, size, key)
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)

            np.save(
                os.path.join(
                    subfolder,
                    'locations.npy'),
                household_locations_region)
            np.save(os.path.join(subfolder, 'size.npy'), household_size_region)
            np.save(os.path.join(subfolder, 'household_types.npy'), household_types_region)
            np.save(os.path.join(subfolder, 'household_incomes.npy'), household_incomes_region)
            
            np.save(
                os.path.join(
                    subfolder,
                    'people_indices.npy'),
                people_indices)
            np.save(os.path.join(subfolder, 'gender.npy'), gender)
            np.save(os.path.join(subfolder, 'age.npy'), age)

def create_gdl_tifs(iso3=None):
    '''This function rasterizes the admin tiffs. This is required to later overlay them with the floodmap to derive where coastal nodes and households should be generated.'''

    try:
        os.makedirs('DataDrive/SLR/GDL')
    except OSError:
        pass
    output_shp = f'DataDrive/SLR/GDL/GDL.shp'
    output_tif = f'DataDrive/SLR/GDL/GDL.tif'
    population_fn = f'DataDrive/POPULATION/GHS_POP_2015.tif'
    if not os.path.exists(output_tif):
        if not os.path.exists(output_shp):
            gdf = gpd.GeoDataFrame.from_file(
                f'DataDrive/GDL/GDL Shapefiles V6.2 large.shp')

            if iso3 is not None:
                gdf = gdf[gdf['iso_code'] == iso3]
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
                gdal.GDT_Int64
            )

def create_shapefiles():
    '''This function generates the shapefiles of the inland nodes and coastal nodes.'''

    tif_file = f'DataDrive/SLR/admin/can_flood_gdl.tif'
    if not os.path.exists(tif_file):
        with rasterio.open(f'DataDrive/SLR/GDL/GDL.tif') as gdl_src:
            gdl_profile = gdl_src.profile
            gdl = gdl_src.read(1)
            gdl_max = np.max(gdl)

        with rasterio.open('DataDrive/SLR/can_flood.tif') as src:
            can_flood = src.read(1)
            can_flood[(np.isnan(can_flood)) | (can_flood < 0.001)] = 0
            can_flood[gdl == gdl_profile['nodata']] = 0
            can_flood = can_flood > 0

        can_flood_gdl = gdl.copy()
        can_flood_gdl[can_flood] += gdl_max
        with rasterio.open(tif_file, 'w', **gdl_profile) as dst:
            dst.write(can_flood_gdl, 1)

    shp_file = tif_file.replace('.tif', '.shp')
    if not os.path.exists(shp_file):
        out = subprocess.call(
            f"python {GDAL_POLYGONIZE} {tif_file} {shp_file}",
            shell=True
        )
        assert out == 0
        assert os.path.exists(shp_file)

    merged_filename = shp_file.replace('.shp', '_merged.shp')
    if not os.path.exists(merged_filename):
        IDs = gpd.GeoDataFrame.from_file(
            f'DataDrive/SLR/GDL/GDL.shp')[f'gdlcode'].to_numpy().astype(str)
        n_IDs = IDs.size - 1

        gdf = gpd.GeoDataFrame.from_file(shp_file)
        ids, geoms, keys = [], [], []
        grouped_by_DN = gdf.groupby(by='DN')
        for DN, geometries in grouped_by_DN:
            if any(geometries.is_valid) == False:
                geometries = geometries.buffer(0)
                geoms.append(unary_union(geometries.geometry))
                geoms.append(unary_union(geometries['geometry']))

            geometries = geometries.buffer(0)
            geoms.append(unary_union(geometries.geometry))

            ids.append(DN)
            if DN > n_IDs:
                keys.append(f"{IDs[DN-n_IDs]}_flood_plain")
            else:
                keys.append(IDs[DN])

        gdf_out = gpd.GeoDataFrame(
            {'ID': ids, 'keys': keys, 'geometry': geoms}, crs="EPSG:4326")
        gdf_out.to_file(merged_filename)

def create_coastline_raster():
    # load shapes
    fn_shape = f'DataDrive/SLR/admin/can_flood_gdl_merged.shp'
    geoms = gpd.read_file(fn_shape)

    # dissolve shapes
    geoms_dissolved = geoms.dissolve()
    outline = geoms_dissolved.boundary
    outline = outline.buffer(0.00833) # include a small buffer to avoid issues with rasterization

    # export to file
    fn_outline = fn_shape.replace('merged', 'outline_v2')
    outline.to_file(fn_outline)

    # rasterize outline
    output_tif = fn_outline.replace('shp', 'tif')
    population_fn = f'DataDrive/POPULATION/GHS_POP_2015.tif'
    
    with rasterio.open(population_fn) as src:
        rasterize_burn(
            fn_outline,
            output_tif,
            src.width,
            src.height,
            4326,
            src.transform.to_gdal(),
            1,
            -1, 
            gdal.GDT_UInt64
        )

if __name__ == '__main__':
    # level = 1
    # print('Creating tifs...')
    # create_gdl_tifs()#, iso3='DEU')  # Optional iso3 argument
    # print('Creating shapes...')
    # create_shapefiles()
    # print('Creating coastline')
    # create_coastline_raster()
    print('creating agents...')
    max_household_size = 15
    # read_synthetic_population_data()
    main(max_household_size)
