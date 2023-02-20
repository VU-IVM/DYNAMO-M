import pandas as pd
import os
import numpy as np
from honeybees.library.mapIO import ArrayReader, NetCDFReader

class Data():
    def __init__(self, model):
        self.model = model
        admin_level = self.model.args.admin_level
        self.data_folder = 'DataDrive'

        # Or use mounted COASTMOVE surfdrive
        if self.model.args.rcp == 'control':
            path = os.path.join('DataDrive', 'EROSION', 'PROCESSED', f'erosion_polynomials_rcp4p5.npy')   
        else:
            path = os.path.join('DataDrive', 'EROSION', 'PROCESSED', f'erosion_polynomials_{self.model.args.rcp}.npy')

        self.shoreline_projections = np.load(path) 
        
        # Load beach IDs and filter on floodzone
        beach_ids = pd.read_csv(os.path.join('DataDrive', 'EROSION', 'PROCESSED', 'beach_ids.csv'))#, index_col='segment_ID')
        self.beach_ids = beach_ids.drop_duplicates(subset='segment_ID').set_index('segment_ID', drop=True)      

        self.coastal_raster = ArrayReader(
            fp=os.path.join(self.data_folder, 'EROSION', 'PROCESSED', 'sandy_beaches.tif'),
            bounds=self.model.bounds
        )

        self.distance_to_coast = ArrayReader(
            fp=os.path.join(self.data_folder, 'AMENITIES', 'dist2coastglobal.tif'),
            bounds=self.model.bounds
        )

        self.hh_income = ArrayReader(
           fp=os.path.join(self.data_folder,'ECONOMY', 'FRA_disposable_hh_income_2016.tif'),
           bounds=self.model.bounds
         )

        self.suitability_arr  = ArrayReader(
            fp=os.path.join(self.data_folder, 'AMENITIES', 'suitability_map.tif'),
            bounds=self.model.bounds)

        # Get start date simulation and load 'closest' pop map
        start_sim = self.model.config['general']['start_time'].year
        GHSL_years = np.array([1990, 2000, 2015])
        year = GHSL_years[np.argmin(abs(GHSL_years - start_sim))]

        print(f'Loading GHSL gridded {year}')

        self.SMOD = ArrayReader(
            fp=os.path.join(self.data_folder, 'POPULATION', 'SMOD', f'GHS_SMOD_E2015_GLOBE_R2022A_54009_1000_V1_0_WGS84.tif'),
            bounds=self.model.bounds
        )

        self.population = ArrayReader(
            fp=os.path.join(self.data_folder, 'POPULATION', f'GHS_POP_{year}.tif'),
            bounds=self.model.bounds
        )

        self.mean_age = ArrayReader(
            fp=os.path.join(self.data_folder, 'POPULATION', f'INSEE_mean_age_2017.tif'),
            bounds=self.model.bounds
        )
        # Load ambient (natural) population change from Omphale 2013-2050
        fp=os.path.join(self.data_folder, 'POPULATION', f'ambient_population_change_gadm_{admin_level}.csv')
        self.nat_pop_change = pd.read_csv(fp, index_col='keys')

        self.HistWorldPopChange = pd.read_excel(r'DataDrive/POPULATION/WPP2019_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.xlsx', sheet_name='ESTIMATES', skiprows=16)
        self.WorldPopChange = pd.read_excel(r'DataDrive/POPULATION/WPP2019_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.xlsx', sheet_name='MEDIUM VARIANT', skiprows=16)
        self.SSP_projections = pd.read_excel(r'DataDrive/POPULATION/iamc_db.xlsx', sheet_name='data', engine='openpyxl')
        
        ## Load inundation maps to dicts
        self.inundation_maps_hist = {}
        self.inundation_maps_2080  = {}
        rts = [1000, 500, 250, 100, 50, 25, 10, 5, 2]
        
        for i in rts:
            fp = os.path.join(self.data_folder,'SLR', 'inundation_maps',  f'inuncoast_historical_wtsub_hist_rp{(str(i).zfill(4))}_0.tif')
            self.inundation_maps_hist[i] = ArrayReader(fp=fp, bounds=self.model.bounds)

        if not self.model.args.rcp == 'control':
            for i in rts:
                fp = os.path.join(self.data_folder,'SLR', 'inundation_maps',  f'inuncoast_{self.model.args.rcp}_wtsub_2080_rp{(str(i).zfill(4))}_0.tif')
                self.inundation_maps_2080[i] = ArrayReader(fp=fp, bounds=self.model.bounds)
        
        # Load expected annual damage factors
        rcp = self.model.args.rcp
        if self.model.args.rcp == 'control':
            rcp = 'rcp4p5'

        fp = os.path.join(self.data_folder, 'SLR', 'inundation_maps', 'PROCESSED', f'annual_ead_{rcp}_fps0010.nc')
        self.ead_map = NetCDFReader(fp, varname='ead', bounds=self.model.bounds)
        # Load stage damage curves        
        self.curves = pd.read_csv(os.path.join(self.data_folder,'SLR',  'damage_curves.csv')).set_index('level')

        # Load amenity functions
        self.coastal_amenity_functions = {}
        self.coastal_amenity_functions['dist2coast'] = pd.read_excel(os.path.join(self.data_folder,'AMENITIES',  'amenity_functions.xlsx'), sheet_name='dist2coast').set_index('distance')
        self.coastal_amenity_functions['beach_amenity'] = pd.read_excel(os.path.join(self.data_folder,'AMENITIES',  'amenity_functions.xlsx'), sheet_name='beach_amenity').set_index('beach_width')
