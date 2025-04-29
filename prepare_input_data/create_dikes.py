import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import os
import pandas as pd
from osgeo import gdal, ogr, osr
import rasterio
from scipy.ndimage import binary_dilation

def move_coastline_landward(export=True):
    # read shape for coastline
    # land_shape = gpd.read_file('DataDrive/OSM/land-polygons-complete-3857/land_polygons.shp')

    # load shape for coastline
    land_shape = gpd.read_file('DataDrive/GDL/GDL.shp')

    # merge all polygons to one 
    land_shape = land_shape.dissolve()

    # convert to mercator
    land_shape = land_shape.to_crs('EPSG:3857')

    # simplify geometry
    land_shape['geometry'] = land_shape.geometry.simplify(100)

    # move coastline landward
    land_shape['geometry'] = land_shape.geometry.buffer(-500)

    # smooth shape
    land_shape['geometry'] = land_shape.geometry.buffer(-5000).buffer(5000)

    if export:
        export_folder = os.path.join('DataDrive', 'coastal_defense')
        os.makedirs(export_folder, exist_ok=True)
        # export outline
        land_shape.boundary.to_file(os.path.join(export_folder, 'shoreline_landward.gpkg'), driver="GPKG")

        # save land file to file
        land_shape.to_file(os.path.join(export_folder, 'land_polygons_landward.gpkg'), driver="GPKG")

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



def rasterize_grid():
    input_shp = os.path.join('DataDrive', 'admin_grids', 'all_grids.gpkg')
    output_tif = os.path.join('DataDrive', 'admin_grids', 'gridded_dike_length.tif')

    population_fn = os.path.join('DataDrive', 'POPULATION', 'GHS_POP_2015.tif')
    with rasterio.open(population_fn) as src:
            rasterize(
                input_shp,
                output_tif,
                src.width,
                src.height,
                4326,
                src.transform.to_gdal(),
                'dike_length',
                -1,
                gdal.GDT_Int32
            )


def create_grid(admin):

    xmin, ymin, xmax, ymax = admin.geometry.bounds

    length = 0.008333333333333333  # 1km in degrees
    wide = 0.008333333333333333  # 1km in degrees

    cols = list(np.arange(xmin, xmax + wide, wide))
    rows = list(np.arange(ymin, ymax + length, length))

    polygons = []
    for x in cols[:-1]:
        for y in rows[:-1]:
            polygons.append(Polygon([(x,y), (x+wide, y), (x+wide, y+length), (x, y+length)]))

    grid = gpd.GeoDataFrame({'geometry':polygons})
    
    # clip grid to admin 
    grid = grid[grid.intersects(admin.geometry.buffer(0.00833))]   

    # convert grid to projected crs
        # homogenize crs (currently all in metric system, not degrees (we want to now distances in m))
    grid = grid.set_crs(epsg=4326)
    grid = grid.to_crs('EPSG:3857')
    # export_folder = os.path.join('DataDrive', 'admin_grids')
    # os.makedirs(export_folder, exist_ok=True)
    # admin_name = admin['keys']
    # grid.to_file(os.path.join(export_folder, f'grid_{admin_name}.gpkg'), driver="GPKG")
    return grid

def length_within_grid(line, grid, admin):
    """Calculate the length of the line segment within each grid cell."""

    grid['dike_length'] = None  # initialize field, float, two decimals
    grid['dike_length'] = grid['dike_length'].astype('float64')
    grid['dike_length'] = grid['dike_length'].round(decimals=2)


    for i, cell in grid.iterrows():
        if line.intersects(cell.geometry).size > 0:
            if line.intersects(cell.geometry)[0]:
                intersection = line.intersection(cell.geometry)
                if intersection.geom_type[0] == 'MultiLineString':
                    length = sum(seg.length for seg in intersection)
                    grid.at[i, 'dike_length'] = length
                else:
                    length = intersection.length
                    grid.at[i, 'dike_length'] = length
            else:
                grid.at[i, 'dike_length'] = 0
        else:
            grid.at[i, 'dike_length'] = 0

    # remove rows where dike length is 0
    grid = grid[grid['dike_length'] > 0]
    
    # export grid with lengths
    export_folder = os.path.join('DataDrive', 'admin_grids')
    os.makedirs(export_folder, exist_ok=True)
    admin_name = admin['keys']
    grid.to_file(os.path.join(export_folder, f'grid_w_length_{admin_name}_length.gpkg'), driver="GPKG")
    print(f'Exported grid with lengths for {admin_name} to {export_folder}')
    return grid

def buffer_urban_area():
    # load SMOD raster as np array
    path = r'DataDrive/POPULATION/SMOD/GHS_SMOD_E2015_GLOBE_R2022A_54009_1000_V1_0_WGS84.tif'
    with rasterio.open(path) as smod_src:
        smod_profile = smod_src.profile
        smod = smod_src.read(1)
    
    # reclassify smod
    smod[smod < 20] = -200
    smod[smod >= 20] = 1

    # save reclassified smod
    with rasterio.open('DataDrive/POPULATION/SMOD/smod_reclassified.tif', 'w', **smod_profile) as dst:
        dst.write(smod, 1)

    # create buffer around urban areas
    # binary smod
    urban_binary = smod == 1
    # create x by x km structuring element
    buffer_size = 15
    structuring_element = np.ones((buffer_size, buffer_size), dtype=bool)
    
    buffered_data = binary_dilation(urban_binary, structure = structuring_element)
    buffered_raster_data = np.where(buffered_data, 1, smod)
    buffered_raster_data[buffered_raster_data != 1] = -200

    with rasterio.open('DataDrive/POPULATION/SMOD/smod_buffered.tif', 'w', **smod_profile) as dst:
        dst.write(buffered_raster_data.astype(rasterio.int16), 1)

def mask_dike_length():
    dike_lenght_fp = 'DataDrive/admin_grids/gridded_dike_length.tif'
    smod_fp = 'DataDrive/POPULATION/SMOD/smod_buffered.tif'

    # mask dike length with urban area
    with rasterio.open(dike_lenght_fp) as dike_src:
        dike = dike_src.read(1)
        dike_profile = dike_src.profile
    with rasterio.open(smod_fp) as smod_src:
        smod = smod_src.read(1)
        smod_profile = smod_src.profile
    
    dike[smod == -200] = -1

    with rasterio.open('DataDrive/admin_grids/gridded_dike_length_masked.tif', 'w', **dike_profile) as dst:
        dst.write(dike, 1)
    


def main():

    # create landward coastline (dike locations)
    move_coastline_landward()

    # load admins
    admins = gpd.read_file('DataDrive/SLR/admin/can_flood_gdl_merged.shp')
    
    # only select coastal floodplains
    admins =  admins[admins['keys'].apply(lambda x: x.endswith('flood_plain'))]
    # load dike line shapes
    dike_locations = gpd.read_file('DataDrive/coastal_defense/shoreline_landward.gpkg')
    admins_reprojected = admins.to_crs('EPSG:3857')

    # initialize empty geodataframe
    grids = gpd.GeoDataFrame()

    for i, admin in admins.iterrows():              
        # create 1km grid for admin 
        grid = create_grid(admin)

        # use grid to intersect with dike lengths        
        # first clip dike location to admin
        dike_locations_clipped = dike_locations[dike_locations.intersects(admins_reprojected.loc[i].geometry)]
        grid = length_within_grid(line=dike_locations_clipped, grid=grid, admin=admin)
        grids = pd.concat([grids, grid], ignore_index=True)

    # export all grids
    grids = gpd.GeoDataFrame(grids)
    export_folder = os.path.join('DataDrive', 'admin_grids')
    os.makedirs(export_folder, exist_ok=True)
    grids.to_file(os.path.join(export_folder, 'all_grids.gpkg'), driver="GPKG")
    rasterize_grid()


if __name__ == '__main__':
    main()
    # buffer_urban_area()
    # mask_dike_length()
    # rasterize_grid()