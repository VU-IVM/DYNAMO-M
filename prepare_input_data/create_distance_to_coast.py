import geopandas as gpd
import os
from shapely.ops import unary_union
import numpy as np
import rasterio
from osgeo import gdal, ogr, osr

def rasterize(input_fn, output_fn, xsize, ysize, projection, gt, attribute, nodatavalue, dtype):
    shp = ogr.Open(input_fn)
    shp_lyr = shp.GetLayer()

    output = gdal.GetDriverByName('GTiff').Create(
        output_fn,
        xsize,
        ysize,
        1,
        dtype,
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
    gdal.RasterizeLayer(output, [1], shp_lyr, options=[f"ATTRIBUTE={attribute}"])

    band = None
    output = None
    shp = None



def create_coastal_segments(iso3 = None):
    '''This function selects from OSM coastline and stores them in folders based the individual floodplains.'''

    # Load shapefile of coastline
    path = os.path.join('DataDrive', 'AMENITIES','coastlines-split-4326', 'lines.shp')
    # path = os.path.join('DataDrive', 'AMENITIES','coastlines-test', 'lines.shp')

    coastline = gpd.read_file(path)
    
    # Load GADM files
    fp =  os.path.join('DataDrive', 'SLR','admin', 'can_flood_gadm_2_merged.shp')
    gadm_merged = gpd.read_file(fp)

    assert coastline.crs == gadm_merged.crs

    if iso3 != None:
        gadm_merged['iso3'] = [keys[:3] for keys in gadm_merged['keys']]
        gadm_filt = gadm_merged[gadm_merged['iso3'] == iso3]

    # Extract floodplains
    floodplains = [key for key in gadm_filt['keys'] if key.endswith('flood_plain')]
    gadm_filt = gadm_filt[gadm_filt['keys'].isin(floodplains)]

    # Small buffer to capture coastline 
    gadm_filt = gadm_filt.buffer(0.2)

    # Iterate over floodplains and select coastal segments within the floodplain
    for i, admin_shape in enumerate(gadm_filt):
        coastline_admin = coastline[coastline.within(admin_shape)]
        assert len(coastline_admin) > 0
        # export to shape
        folder = os.path.join('DataDrive', 'AMENITIES', 'PROCESSED', floodplains[i])
        if not os.path.exists(folder):
            os.mkdir(folder)
        coastline_admin.to_file(os.path.join(folder, 'coastline.shp'))

    
def create_cells_adjecent_to_coast():
    '''This function rasterizes the masked shoreline projections data. It then creates a raster in which 1km cells that are adjecent to the coast have value 1, other cells have value 0.
    '''
    # Load population raster for extent, crs and cellsize
    population_fn =  os.path.join('DataDrive', 'POPULATION', 'GHS_POP_2015.tif')
    
    
    # Load masked shoreline projections 
    path = os.path.join('DataDrive', 'EROSION', 'PROCESSED', 'shoreline_projections_2050_RCP45.npy')
    
    if not os.path.exists(path):
        raise ValueError('Clipped shoreline projections not found. First run prepare_input_data.py')
    
    shoreline_projections = np.load(path)

    # transform to points
    location_gpd = gpd.GeoDataFrame(np.full(shoreline_projections.shape[0], 1, dtype=np.int16), geometry=gpd.points_from_xy(shoreline_projections[:, 1], shoreline_projections[:, 0])).set_crs(epsg = 4326)
    location_gpd.columns = [str(column) for column in location_gpd.columns]
    input_fn = os.path.join('DataDrive', 'EROSION', 'PROCESSED', 'sandy_beaches.shp')
    location_gpd.to_file(input_fn)
    output_fn = os.path.join('DataDrive', 'EROSION', 'PROCESSED', 'sandy_beaches.tif')

    with rasterio.open(population_fn) as src:
                rasterize(
                    input_fn,
                    output_fn,
                    src.width,
                    src.height,
                    4326,
                    src.transform.to_gdal(),
                    '0',
                    -1,
                    gdal.GDT_Int32
                )



if __name__ == '__main__':
    # create_coastal_segments(iso3='FRA')
    create_cells_adjecent_to_coast()