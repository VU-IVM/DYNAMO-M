import geopandas as gpd
import rasterio
import os
from osgeo import gdal, ogr, osr
import numpy as np


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


def rasterize_flopros_global():
    # load FLOPROS flood protection standards
    input_fn = os.path.join(
        'DataDrive',
        'SLR',
        'FLOPROS_shp_V1',
        'FLOPROS_shp_V1.shp')
    adjusted_fn = os.path.join(
        'DataDrive',
        'SLR',
        'FLOPROS_shp_V1',
        'FLOPROS_adjusted.shp')

    flopros = gpd.read_file(input_fn)

    # take maximum fps found in policy or design layer
    flopros['adj_fpd_co'] = np.maximum(
        flopros['DL_Min_Co'], flopros['PL_Min_Co'])
    flopros.to_file(adjusted_fn)

    # Load population raster for extent, crs and cellsize and rasterize
    # coastal segemtns
    population_fn = os.path.join('DataDrive', 'POPULATION', 'GHS_POP_2015.tif')
    output_fn = os.path.join('DataDrive', 'SLR', 'FLOPROS_coastal.tif')

    with rasterio.open(population_fn) as src:
        rasterize(
            adjusted_fn,
            output_fn,
            src.width,
            src.height,
            4326,
            src.transform.to_gdal(),
            'ModL_Riv',
            -1,
            gdal.GDT_Int32
        )


if __name__ == '__main__':
    rasterize_flopros_global()
