'''In this script all data are loaded into the data class. 
We do not include the actual data in this repository, since it can be downloaded though official sources stated in the publication.'''

import pandas as pd
import os
import numpy as np
from dateutil.relativedelta import relativedelta
from honeybees.library.mapIO import NetCDFReader, ArrayReader
from honeybees.library.raster import write_to_array

class Data():
    def __init__(self, model):
        self.model = model
        admin_level = self.model.args.admin_level
        self.data_folder = 'DataDrive'

        coastal_admins = pd.read_csv(os.path.join(self.data_folder, 'SLR', 'admin', 'coastal_admin.csv'))['keys']
        self.coastal_admins = np.array([coastal_admins])

        # Or use mounted COASTMOVE surfdrive
        # self.data_folder = 'X:\Shared\COASTMOVE\DataDrive' 
        # Translate GADM to NUTS1

        self.dist_to_coast = ArrayReader(
            fp=os.path.join(self.data_folder, 'AMENITIES', 'dist2coastglobal.tif'),
            bounds=self.model.bounds
        )

        self.unemployment_rate  = ArrayReader(
            fp=os.path.join(self.data_folder, 'ECONOMY', 'INSEE.tif'),
            bounds=self.model.bounds
        )

        self.hh_income = ArrayReader(
           fp=os.path.join(self.data_folder,'SLR', 'FRA_disposable_hh_income_2008.tif'),
           bounds=self.model.bounds
         )
       
        self.amenity_value  = ArrayReader(
            fp=os.path.join(self.data_folder, 'AMENITIES', 'amenity_total.tif'),
            bounds=model.bounds 
        )      
    
        suitability  = ArrayReader(
            fp=os.path.join(self.data_folder, 'AMENITIES', 'suitability_map.tif'),
            bounds=[-180, 180, -90, 90])
        self.suitability_arr = suitability.get_data_array()


        # Get start date simulation and load 'closest' pop map
        start_sim = self.model.config['general']['start_time'].year
        GHSL_years = np.array([1990, 2000, 2015])
        year = GHSL_years[np.argmin(abs(GHSL_years - start_sim))]

        print(f'Loading GHSL gridded {year}')

        self.population = ArrayReader(
            fp=os.path.join(self.data_folder, 'POPULATION', f'GHS_POP_{year}.tif'),
            bounds=self.model.bounds
        )
        # Load ambient (natural) population change from Omphale 2013-2050
        fp=os.path.join(self.data_folder, 'POPULATION', f'ambient_population_change_gadm_{admin_level}.csv')
        self.nat_pop_change = pd.read_csv(fp, index_col='keys')

        self.HistWorldPopChange = pd.read_excel(r'DataDrive/POPULATION/WPP2019_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.xlsx', sheet_name='ESTIMATES', skiprows=16)
        self.WorldPopChange = pd.read_excel(r'DataDrive/POPULATION/WPP2019_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.xlsx', sheet_name='MEDIUM VARIANT', skiprows=16)

        # Load files for distribution of international migrants
        fp=os.path.join(self.data_folder, 'POPULATION', f'international_migration_gadm_{admin_level}.csv')
        try:
            self.international_migration_dist = pd.read_csv(fp, index_col='keys')
        except:
            self.international_migration_dist = None
            
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
           
        # Load stage damage curves        
        self.curves = pd.read_excel(os.path.join(self.data_folder,'SLR',  'curves_2010.xlsx'), nrows =1)
        self.curves= self.curves.transpose()
        self.curves = self.curves.reset_index()
        
        self.curves_dryproof_1m = pd.read_excel(os.path.join(self.data_folder,'SLR', 'curves_2010_dryproof_1m.xlsx'), nrows =1)
        self.curves_dryproof_1m= self.curves_dryproof_1m.transpose()
        self.curves_dryproof_1m = self.curves_dryproof_1m.reset_index()
