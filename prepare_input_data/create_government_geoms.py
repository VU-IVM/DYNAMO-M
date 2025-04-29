import numpy as np
import geopandas as gpd
import pandas as pd
import os
import rasterio
from osgeo import gdal, ogr, osr

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

def rasterize_geoms(file='government_geoms.gpkg'):
    input_shp = os.path.join('DataDrive', 'government_geoms', file)
    output_tif = os.path.join('DataDrive', 'government_geoms', file.replace('.gpkg', '.tif'))

    population_fn = os.path.join('DataDrive', 'POPULATION', 'GHS_POP_2015.tif')
    with rasterio.open(population_fn) as src:
            rasterize(
                input_shp,
                output_tif,
                src.width,
                src.height,
                4326,
                src.transform.to_gdal(),
                'idx',
                -1,
                gdal.GDT_Int32
            )

def create_geoms():
    
    # shape size limits (size in degrees^2)
    limit = 0.12

    # get countries in model
    agents_fn = r'/scistor/ivm/ltf200/COASTMOVE/prepare_input_data/DataDrive/SLR/households_gdl_2015/xxlarge'
    countries = np.unique([region[:3] for region in os.listdir(agents_fn)])

    
    # load coastal floodplain
    print('Loading and filtering coastal floodplain...')
    coastal_floodplain = gpd.read_file("DataDrive/SLR/admin/can_flood_gdl_merged.shp")
    # filter to only include floodplains
    coastal_floodplain = coastal_floodplain[coastal_floodplain['keys'].str.endswith('flood_plain')]
    
    # Load the GDL geoms
    print('Loading GADM geometries...')
    gadm_1 = gpd.read_file("DataDrive/GADM/gadm41_1.gpkg")
    gadm_2 = gpd.read_file("DataDrive/GADM/gadm41_2.gpkg")
    print('Loaded GADM geometries')
    # gadm_1 = gadm_1[gadm_1['GID_0'].isin(countries)]
    
    coastal_gadm = []

    # iterate over countries
    for country in countries:
        print(f'Processing {country}')
        try:
            # only keep the geometries that intersect the coastal floodplain
            country_shapes = gadm_2[gadm_2['GID_0'] == country]
            level = 2
            names = list(country_shapes['GID_2'])
            if (country_shapes.geometry.area.median() < limit and country != 'GBR') or len(country_shapes) == 0:
                # take gadm 1 if shape is smaller than limit
                country_shapes = gadm_1[gadm_1['GID_0'] == country]
                level = 1
                names = country_shapes['GID_1']
            
            country_shapes = country_shapes.reset_index()
            # only use geometry and country
            country_shapes = country_shapes[['GID_0', 'geometry']]
            country_shapes['keys'] = list(names)
            # clip coastal floodplain
            print('Clipping coastal floodplain...')
            coastal_floodplain_country = coastal_floodplain[coastal_floodplain['keys'].str.startswith(country)]
            
            # only select the geometries that intersect the coastal floodplain
            print('Intersecting geometries with coastal floodplain...')
            country_shapes = country_shapes[country_shapes.intersects(coastal_floodplain_country.unary_union)]
            
            # add level to country_shapes
            country_shapes['LEVEL'] = np.full(len(country_shapes), level, np.int16)

            # store
            coastal_gadm.append(country_shapes)

            # export country shapes
            export_folder = os.path.join('DataDrive', 'government_geoms', 'individual_countries')
            os.makedirs(export_folder, exist_ok=True)
            country_shapes.to_file(os.path.join(export_folder, f'{country}_geoms.gpkg'), driver="GPKG")
            print(f'Exported {country} to {export_folder}')
        except:
            print(f'Failed to process {country}, now using GDL region')
            # load gdl region
            gdl = gpd.read_file("DataDrive/GDL/GDL.shp")
            gdl = gdl[gdl['iso_code'] == country]
            gdl['keys'] = gdl['gdlcode']
            gdl['LEVEL'] = np.full(len(gdl), -1, np.int16)
            gdl['GID_0'] = country

            # store
            coastal_gadm.append(country_shapes[['GID_0', 'geometry', 'keys', 'LEVEL']])

    # merge list into single geodataframe
    all_coastal = pd.concat(coastal_gadm, ignore_index=True)
    all_coastal = gpd.GeoDataFrame(all_coastal)
    all_coastal['idx'] = np.arange(len(all_coastal))
    export_folder = os.path.join('DataDrive', 'government_geoms')
    os.makedirs(export_folder, exist_ok=True)
    all_coastal.to_file(os.path.join(export_folder, 'government_geoms.gpkg'), driver="GPKG")

if __name__ == "__main__":
    create_geoms()
    rasterize_geoms()