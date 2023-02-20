'''The functions in this script are used to interpolate annual flood depths and to create expected annual damages per grid cell for 2015-2080. Results are store in netcdf format.'''
import numpy as np
import xarray as xr
import rasterio
import matplotlib.pyplot as plt
import os
import requests
import subprocess
import pandas as pd
import pickle
from scipy import interpolate
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="{asctime} {levelname:<8} {message}",
    style = '{',
    filename='%slog' % __file__[:-2],
    filemode='w'
)

class filled_floodmap:
    def __init__(self, dimensions, data):
        self.dimensions = dimensions
        self.hazard_map = np.full(dimensions, -1, np.float32)


def download_flood_maps(percentile=95, fps = 0):  
    # Store in 
    datafolder = os.path.join('DataDrive', 'SLR', 'Aqueduct')
    if not os.path.exists(datafolder):
        os.makedirs(datafolder)
    
    # download baseline
    file_base = 'inuncoast_historical_nosub_hist_rp'
    rts = [2, 5, 10, 25, 50, 100, 250, 500, 1000]
    url_base = 'http://wri-projects.s3.amazonaws.com/AqueductFloodTool/download/v2/'

    for rt in rts:
        file_name = file_base + str(rt).zfill(4) + '_0.tif'
        url = url_base + file_name
        response = requests.get(url)
        open(os.path.join(datafolder, file_name), 'wb').write(response.content)
    
    # download clim scenarios
    for rcp in ['rcp4p5', 'rcp8p5']:
        for year in ['2080']: #['2030', '2050', '2080']:
            file_base = f'inuncoast_{rcp}_wtsub_{year}_rp'
            for rt in rts:
                file_name = file_base + str(rt).zfill(4) + '_0.tif'
                url = url_base + file_name
                response = requests.get(url)
                open(os.path.join(datafolder, file_name), 'wb').write(response.content)


def create_netcdf_for_template():
    # convert tif to netcdf to get lat lon attributes
    output_folder = os.path.join('DataDrive', 'SLR', 'inundation_maps')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    input_file = os.path.join('DataDrive', 'SLR', 'Aqueduct', 'inuncoast_historical_nosub_hist_rp0002_0.tif')
    output_file =  os.path.join(output_folder, 'inun_2015_rp0002_0.nc')
    out = subprocess.call(
            f'gdal_translate -of NetCDF {input_file} {output_file}',
            shell=True)


def interpolate_annual_flood_depths(rcp='rcp4p5', fps=0):
    # set datafolder
    logging.debug('Executing interpolate_annual_flood_depths...')
    datafolder = os.path.join('DataDrive', 'SLR', 'Aqueduct')
    processed_folder = os.path.join('DataDrive', 'SLR', 'Aqueduct', 'PROCESSED')
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
    # Iterate of scenarios
    annual_maps = {}
        
    annual_maps[rcp] = {}
    rts = np.array([1000, 500, 250, 100, 50, 25, 10, 5, 2])
    rts = rts[rts >= fps]

    for rt in rts:
        logging.info(f'Calculating annual floodmaps for rp {rt}...')
        # load hist 
        src = rasterio.open(os.path.join(datafolder, f'inuncoast_historical_nosub_hist_rp{str(rt).zfill(4)}_0.tif'))
        array_hist = src.read(1)
        # load future 
        src = rasterio.open(os.path.join(datafolder, f'inuncoast_{rcp}_wtsub_2080_rp{str(rt).zfill(4)}_0.tif'))
        array_future =  src.read(1)

        # mask hist'
        mask = np.where(array_future > 0)

        # Assert there is data grid cells that are flooded in the future (NoData val = -9999)
        assert (array_hist[mask] > -1).all()

        # Derive annual increment (now uniform rate, replace this with an increasing rate at some point)
        increment = np.zeros(array_hist.shape, dtype=np.float32)
        increment[mask] = (array_future[mask] - array_hist[mask]) / 65

        # create annual inundation maps
        
        annual_maps[rcp][rt] = {}
        annual_maps[rcp][rt][2015] = {}
        annual_maps[rcp][rt][2015]['inundation'] = array_hist[mask]
        annual_maps[rcp][rt][2015]['mask'] = mask

        for i in np.arange(65):
            array_hist[mask] += increment[mask]
            annual_maps[rcp][rt][(2016 + i)] = {}
            annual_maps[rcp][rt][(2016 + i)]['inundation'] = array_hist[mask]
            annual_maps[rcp][rt][(2016 + i)]['mask'] = mask

    annual_maps['shape'] = array_hist.shape
    logging.debug('All annual floodmaps created')

    # Store maps
    fp = os.path.join('DataDrive', 'SLR', 'inundation_maps', f'annual_flood_maps_{rcp}_fps{str(fps).zfill(4)}.pickle')
    f = open(fp,"wb")
    pickle.dump(annual_maps,f)
    f.close()
    logging.debug('All annual floodmaps stored')

    del annual_maps

def integrate_expected_damages(
    annual_maps_fp,
    curves_fp=os.path.join('DataDrive', 'damage_curves.csv'),
    rcp='rcp4p5',
    fps=0,
    ):
    logging.info('Executing integrate_expected_damages...')
    # load curves 
    with open(annual_maps_fp, 'rb') as pickle_file:
        annual_maps = pickle.load(pickle_file)

    curves = pd.read_csv(curves_fp, index_col='level')['baseline']
    depths = np.array(curves.index)
    damage_factor = np.array(curves)
    interpolater_damages = interpolate.interp1d(x = depths, y = damage_factor)
    years = np.arange(2015, 2081)
    damage_maps = {}
    
    for year in years:
        damage_maps[year] = {}

    logging.debug(f'running {rcp}...')
    expected_annual_damages = {}
    shape_map = annual_maps['shape']
    expected_annual_damages['shape'] = shape_map

    for rp in annual_maps[rcp]:
        for year in years:# annual_maps[rcp][rp]:
            damage_maps[year][rp] = {}
            annual_maps[rcp][rp][year]['inundation'] = np.maximum(annual_maps[rcp][rp][year]['inundation'], np.min(curves.index))
            annual_maps[rcp][rp][year]['inundation'] = np.minimum(annual_maps[rcp][rp][year]['inundation'], np.max(curves.index))
            damage_maps[year][rp]['damage_factor']  = interpolater_damages(annual_maps[rcp][rp][year]['inundation'])
            damage_maps[year][rp]['mask'] = annual_maps[rcp][rp][year]['mask']
    del annual_maps

    # Now create an expected annual damage factor for each year  
    for year in years: #damage_maps:
        expected_annual_damages[year] = {}
        # initiate numpy array for filling with damage data
        array_for_integration = np.full((len(damage_maps[year]), damage_maps[year][1000]['damage_factor'].size), -1, dtype=np.float32)

        # fist create map with zeros and make sure that the dimensions and nodata values of different floodmaps correspond
        probabilities = [1/rp for rp in damage_maps[year]]
        for i, rp in enumerate(damage_maps[year]):
            fill_map = np.zeros(shape_map, dtype=np.float32)
            fill_map[damage_maps[year][rp]['mask']] = damage_maps[year][rp]['damage_factor']
            filled_floodmap = fill_map[damage_maps[year][1000]['mask']]
            assert filled_floodmap.size == array_for_integration.shape[1]
            array_for_integration[i] = filled_floodmap
            del fill_map, filled_floodmap
        # calculate expected annual damage (factor)
        ead = np.trapz(y = array_for_integration, x = probabilities, axis = 0)
        
        # store annual ead in dict
        expected_annual_damages[year]['data'] = ead
        expected_annual_damages[year]['mask'] = damage_maps[year][1000]['mask']
        logging.debug(f'Current year {year}')
    
    # Store as pickle
    logging.debug('done, saving files...')
    fp = os.path.join('DataDrive', 'SLR', 'inundation_maps', f'annual_expected_damages_{rcp}_fps{str(fps).zfill(4)}.pickle')
    f = open(fp,"wb")
    pickle.dump(expected_annual_damages, f)
    f.close()

def export_dict_to_netcdf(ead_map_fp, rcp, fps):
    logging.info('Executing export_dict_to_netcdf...')
    # load nc data
    fn = os.path.join('DataDrive', 'SLR', 'inundation_maps', 'inun_2015_rp0002_0.nc')
    nc_template = xr.open_dataset(fn)

    # Create time dimension array
    years = np.arange(2015, 2081)
    time = [datetime(year=year, month=1, day=1) for year in years]

    with open(ead_map_fp, 'rb') as pickle_file:
        ead_maps = pickle.load(pickle_file)

    # preallocate array with time dimension
    dims = (years.size, ead_maps['shape'][0], ead_maps['shape'][1])
    
    logging.debug('creating emtpy frame...')
    export_array = np.full(dims, np.nan, dtype = np.float32)

    expected_annual_damages_export = xr.Dataset()
    logging.debug('constructing dataframe...')
    
    for i, year in enumerate(years):
        # Fill map 
        logging.debug(f'current year in df: {year}')
        export_array[i, :][ead_maps[year]['mask']] = ead_maps[year]['data']
        export_array[i, :] = np.flipud(export_array[i, :])

    coords = nc_template.coords
    coords['time'] = time

    logging.debug('creating xarray...')
    map_export = xr.DataArray(export_array, coords=coords, dims=['time', 'lat', 'lon'])
    expected_annual_damages_export['ead'] = map_export
    expected_annual_damages_export.assign({'crs':nc_template.crs}) 
    encoding={}

    logging.debug('compressing and exporting data ...')
    for variable in list(expected_annual_damages_export.variables):
        encoding[variable] = {"zlib": True, "complevel": 9}

    outfile = os.path.join('DataDrive', 'SLR', 'inundation_maps', f'annual_ead_{rcp}_fps{str(fps).zfill(4)}.nc')
    expected_annual_damages_export.to_netcdf(outfile, encoding=encoding)

if __name__ == '__main__':
    # download_flood_maps()
    # create_netcdf_for_template()   
    
    for rcp in ['rcp4p5', 'rcp8p5']:
        for fps in [2, 10]:
            
            interpolate_annual_flood_depths(rcp=rcp, fps=fps)
            
            annual_maps_fp = os.path.join('DataDrive', 'SLR', 'inundation_maps', f'annual_flood_maps_{rcp}_fps{str(fps).zfill(4)}.pickle')
            integrate_expected_damages(annual_maps_fp, fps=fps, rcp=rcp)
            
            ead_map_fp = os.path.join('DataDrive', 'SLR', 'inundation_maps', f'annual_expected_damages_{rcp}_fps{str(fps).zfill(4)}.pickle')
            export_dict_to_netcdf(ead_map_fp, rcp=rcp, fps=fps)
