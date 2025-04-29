import numpy as np
import os
import pandas as pd
import geopandas as gpd
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


def create_mean_to_median_ratios(datafolder):
    '''Data downloaded from: https://stats.oecd.org/Index.aspx?DataSetCode=IDD'''

    income_db_fp = os.path.join(datafolder, 'OECD', 'income_distibution_db.csv')
    # load data
    income_db = pd.read_csv(income_db_fp)

    # iterate over countries in dataset
    countries_db = np.unique(income_db['LOCATION'])
    mean_disp_income = []
    median_disp_income = []
    years = []
    for country in countries_db:
        # subset
        subset_inc = income_db.loc[
            (income_db['LOCATION'] == country) & 
            (income_db['MEASURE'] == 'INCCTOTAL') & 
            (income_db['AGE'] == 'TOT') &
            (income_db['METHODO'] == 'METH2012')][['TIME', 'Value']]
        # subset median
        median_subset_inc = income_db.loc[
            (income_db['LOCATION'] == country) & 
            (income_db['MEASURE'] == 'MEDIANC') & 
            (income_db['AGE'] == 'TOT') &
            (income_db['METHODO'] == 'METH2012')][['TIME', 'Value']]
        
        # find year that is present in both datasets
        yr_arr = np.arange(2030, 2000, -1)
        for yr in yr_arr:
            if yr in list(subset_inc['TIME']) and yr in list(median_subset_inc['TIME']):  
                mean_disp_income.append(subset_inc.loc[(subset_inc['TIME'] == yr)]['Value'].iloc[0])
                median_disp_income.append(median_subset_inc.loc[(median_subset_inc['TIME'] == yr)]['Value'].iloc[0])
                years.append(yr)
                break
    # construct dataset
    mean_median_ratio = np.array(mean_disp_income)/ np.array(median_disp_income)
    export_df = pd.DataFrame(
        {
            'country': countries_db,
            'year': years,
            'mean_inc': mean_disp_income,
            'median_inc': median_disp_income,
            'mean_median_ratio': mean_median_ratio})
    # export
    fp_export = os.path.join(datafolder, 'PROCESSED', 'mean_median_OECD.csv') 
    export_df.to_csv(fp_export, index=False)

def prepare_income_data_synthpop(path):
    fp = os.path.join(path, 'summary.csv')  
    income_data = pd.read_csv(fp)[['GLDcode', 'average_income_region']].dropna()
    save_to = os.path.join(path, 'gdl_income_2015.csv')
    if not os.path.exists(os.path.dirname(save_to)):
        os.makedirs(os.path.dirname(save_to))
    income_data.columns = ['GDLcode', 'DISP_INCOME']
    income_data.to_csv(save_to, index=False)
    return save_to


def process_income_data(path):
    # load income csv
    fp = os.path.join(path, 'GDL-Log-Gross-National-Income-per-capita-in-1000-US-Dollars-(2011-PPP)-data.csv')
    income_data = pd.read_csv(fp)[['GDLCODE', '2015']].dropna()
    income_data['INCOME'] = np.exp(income_data['2015']).astype(np.int32)
    income_data['DISP_INCOME'] = np.round(income_data['INCOME'] * 0.5).astype(np.int32)     # scale income to known household income in France
    income_data = income_data[['GDLCODE', 'INCOME', 'DISP_INCOME']]
    save_to = os.path.join(path, 'PROCESSED', 'gdl_income_2015.csv')
    if not os.path.exists(os.path.dirname(save_to)):
        os.makedirs(os.path.dirname(save_to))
    income_data.to_csv(save_to, index=False)
    return save_to

def rasterize_income_data(processed_income_fp):
    output_shp = os.path.join('DataDrive', 'ECONOMY', 'PROCESSED', 'gdl_income_2015.shp')
    output_tif = os.path.join('DataDrive', 'ECONOMY', 'PROCESSED', 'gdl_income_2015.tif')
    population_fn = os.path.join('DataDrive', 'POPULATION', 'GHS_POP_2015.tif')
    gdf = gpd.GeoDataFrame.from_file(
        f'DataDrive/SLR/GDL/GDL.shp')[['gdlcode', 'geometry']]
    
    processed_income = pd.read_csv(processed_income_fp)
    # join income 
    
    merged_gdf = gdf.merge(processed_income, left_on='gdlcode', right_on='GDLcode')[['gdlcode', 'DISP_INCOME', 'geometry']]
    merged_gdf.to_file(output_shp)
        
    with rasterio.open(population_fn) as src:
        rasterize(
            output_shp,
            output_tif,
            src.width,
            src.height,
            4326,
            src.transform.to_gdal(),
            'DISP_INCOM',
            -1,
            gdal.GDT_Int32
        )





if __name__ == "__main__":
    create_mean_to_median_ratios(datafolder=os.path.join('DataDrive', 'ECONOMY'))
    
    # processed_income_fp = prepare_income_data_synthpop(os.path.join('DataDrive', 'AGENTS', 'PROCESSED'))
    # processed_income_fp = process_income_data(os.path.join('DataDrive', 'ECONOMY'))
    # rasterize_income_data(processed_income_fp)
