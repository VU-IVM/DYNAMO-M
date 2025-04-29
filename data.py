import pandas as pd
import os
import numpy as np
from honeybees.library.mapIO import MapReader
from scipy import interpolate


class Data():
    def __init__(self, model):

        self.model = model
        self.data_folder = 'DataDrive' # r'U:\COASTMOVE_data\DataDrive'

        ############################
        # Load GADM tiff for gov #
        ############################

        self.government_gadm = MapReader(
            fp=os.path.join(self.data_folder, 'GADM', 'government_geoms_merged.tif'),
            xmin=self.model.xmin,
            ymin=self.model.ymin,
            xmax=self.model.xmax,
            ymax=self.model.ymax,
        )

        ##########################
        #### Load UNSD (M49) #####
        ##########################
        path = os.path.join(self.data_folder, 'UNSD — Methodology.csv')
        self.UNSD_M49 = pd.read_csv(path, sep = ';', index_col='ISO-alpha3 Code')       
        
        ######################################
        #### Load WIID distribution data #####
        ######################################
        path = os.path.join(self.data_folder, 'ECONOMY', 'PROCESSED', 'mean_median_WIID.csv')
        self.UN_WIID = pd.read_csv(path, index_col=0)
        
        ######################################
        #### Load flood risk information #####
        ######################################

        # load rasterized flopros
        self.flopros = MapReader(
            fp=os.path.join(
                self.data_folder,
                'SLR',
                'PROCESSED',
                f'FLOPROS_coastal.tif'),
            xmin=self.model.xmin,
            ymin=self.model.ymin,
            xmax=self.model.xmax,
            ymax=self.model.ymax,
        )

        # load water levels
        self.load_water_levels()

        # load damage_functions
        # Load stage damage curves
        curves_baseline = pd.read_excel(os.path.join(
            self.data_folder, 'SLR', 'damage_functions.xlsx'), sheet_name='baseline', index_col='Flood depth')

        curves_dryproof = pd.read_excel(os.path.join(
            self.data_folder, 'SLR', 'damage_functions.xlsx'), sheet_name='dryproof_1m', index_col='Flood depth')

        curves_industrial = pd.read_excel(os.path.join(
            self.data_folder, 'SLR', 'damage_functions.xlsx'), sheet_name='industrial', index_col='Flood depth')

        curves_commercial = pd.read_excel(os.path.join(
            self.data_folder, 'SLR', 'damage_functions.xlsx'), sheet_name='commercial', index_col='Flood depth')


        # initiate dictionary
        self.dam_func = {}
        self.dam_func_dryproof_1m = {}
        self.dam_func_industrial = {}
        self.dam_func_commercial = {}

        # iterate over regions and store curves in dictionary
        for region in curves_baseline.columns:
            damage_inter = interpolate.interp1d(
                curves_baseline.index, curves_baseline[region])
            damage_dryproof_1m_inter = interpolate.interp1d(
                curves_dryproof.index, curves_dryproof[region])
            damage_industrial_inter = interpolate.interp1d(
                curves_industrial.index, curves_industrial[region])
            damage_commercial_inter = interpolate.interp1d(
                curves_commercial.index, curves_commercial[region])

            # get array of depths
            depths = np.arange(0, max(curves_baseline.index)+0.01, 0.01)
            # each cm inundation corresponds to the index in these arrays.
            self.dam_func[region] = damage_inter(depths)
            self.dam_func[region][0] = 0 # assert that zero depth corresponds to zero damage (USA fix)
            self.dam_func_dryproof_1m[region] = damage_dryproof_1m_inter(depths)
            self.dam_func_dryproof_1m[region][0] = 0 # assert that zero depth corresponds to zero damage (USA fix)  
            self.dam_func_industrial[region] = damage_industrial_inter(depths)
            self.dam_func_commercial[region] = damage_commercial_inter(depths)

        # load max damages for countries
        self.max_damage_data = pd.read_csv(os.path.join(
            self.data_folder, 'SLR', 'max_damage_countries.csv')).set_index('iso3_code')


        # load adaptation costs for countries
        self.adaptation_cost = pd.read_csv(os.path.join(
            self.data_folder, 'ECONOMY', 'PROCESSED', 'scaled_adaptation_cost.csv')).set_index('iso3_code')

        # load cost of elevating a sea dike (eur/m/km)
        self.dike_elevation_cost = pd.read_csv(os.path.join(
            self.data_folder, 'ECONOMY', 'PROCESSED', 'scaled_dike_elevation_cost.csv')).set_index('iso3_code')

        # load cost of maintaining a sea dike (eur/km)
        self.dike_maintenance_cost = pd.read_csv(os.path.join(
            self.data_folder, 'ECONOMY', 'PROCESSED', 'scaled_dike_maintenance_cost.csv')).set_index('iso3_code')

        # load rasterized outline for dike locations
        if self.model.args.subdivision == 'GADM':
            raise NotImplementedError('GADM not supported anymore')
            fp = os.path.join(self.data_folder, 'SLR', 'admin', 'can_flood_gadm_1_outline.tif')
        elif self.model.args.subdivision == 'GDL':
            fp = os.path.join(self.data_folder, 'SLR', 'admin', 'gridded_dike_length.tif')

        self.coastal_dike_lengths = MapReader(
            fp=fp,
            xmin=self.model.xmin,
            ymin=self.model.ymin,
            xmax=self.model.xmax,
            ymax=self.model.ymax
        )
        
        #####################################################################
        #### Load files related to coastal amenities and shoreline change####
        #####################################################################

        # load shoreline trend csv. Also includes lat lon segments and DoC. For
        # now load and entire world.
        self.shoreline_change_trend = MapReader(
            fp=os.path.join(
                self.data_folder,
                'EROSION',
                'PROCESSED',
                f'rasterized_changerate_world_adjusted.tif'),
            xmin=self.model.xmin,
            ymin=self.model.ymin,
            xmax=self.model.xmax,
            ymax=self.model.ymax
        )
        if self.model.args.rcp != 'control':
            self.shoreline_loss_2050 =  MapReader(
                fp=os.path.join(
                    self.data_folder,
                    'EROSION',
                    'PROCESSED',
                    f'rasterized_beach_loss_2050_world_{self.model.args.rcp}_adjusted.tif'),
                xmin=self.model.xmin,
                ymin=self.model.ymin,
                xmax=self.model.xmax,
                ymax=self.model.ymax
            )

            self.shoreline_loss_2100 =  MapReader(
                fp=os.path.join(
                    self.data_folder,
                    'EROSION',
                    'PROCESSED',
                    f'rasterized_beach_loss_2100_world_{self.model.args.rcp}_adjusted.tif'),
                xmin=self.model.xmin,
                ymin=self.model.ymin,
                xmax=self.model.xmax,
                ymax=self.model.ymax
            )


        # load raster of depth of closures
        self.depth_of_closure = MapReader(
            fp=os.path.join(
                self.data_folder,
                'EROSION',
                'PROCESSED',
                f'rasterized_doc_world_adjusted.tif'),
            xmin=self.model.xmin,
            ymin=self.model.ymin,
            xmax=self.model.xmax,
            ymax=self.model.ymax
        )
        # Load boolean raster indicating wheter household location is close to
        # a beach
        self.sandy_beach_cells = MapReader(
            fp=os.path.join(
                self.data_folder,
                'EROSION',
                'PROCESSED',
                f'sandy_beach_cells_world_adjusted.tif'),
                xmin=self.model.xmin,
                ymin=self.model.ymin,
                xmax=self.model.xmax,
                ymax=self.model.ymax
            )

        # Load rasterized euclidian distance to coastline
        self.distance_to_coast = MapReader(
            fp=os.path.join(self.data_folder, 'AMENITIES',
                            'distance_to_coast_gdl.tif'),
            xmin=self.model.xmin,
            ymin=self.model.ymin,
            xmax=self.model.xmax,
            ymax=self.model.ymax
        )


        # Load amenity functions
        self.coastal_amenity_functions = {}

        self.coastal_amenity_functions['dist2coast'] = pd.read_excel(
            os.path.join(
                self.data_folder,
                'AMENITIES',
                'amenity_functions.xlsx'),
            sheet_name='dist2coast').set_index('distance')

        self.coastal_amenity_functions['beach_amenity'] = pd.read_excel(
            os.path.join(
                self.data_folder,
                'AMENITIES',
                'amenity_functions.xlsx'),
            sheet_name='beach_amenity').set_index('beach_width')

        ################################################################
        #### Load files related to population and population change ####
        ################################################################

        # load SSP projections
        
        self.national_GDP = pd.read_csv(
            os.path.join(self.data_folder, 'ECONOMY', 'GDP_2015_USD_WorldBank.csv'),
            skiprows=4,
            index_col='Country Code'
        )[['2015']]


        # download from
        # https://secure.iiasa.ac.at/web-apps/ene/SspDb/dsd?Action=htmlpage&page=30
        self.SSP_GDP_projections = pd.read_excel(
            os.path.join(self.data_folder, 'SSPs', 'GDP_PPP_per_capita_filled.xlsx'),
            sheet_name='data',
            engine='openpyxl')

        self.SSP_population_projections = pd.read_excel(
            os.path.join(self.data_folder, 'SSPs', 'population_filled.xlsx'),
            sheet_name='data',
            engine='openpyxl')

        # load and clean income sheet World Bank
        fp = os.path.join(self.data_folder, 'ECONOMY', 'adjusted_net_national_income_pc_WoldBank_filled.csv')
        df = pd.read_csv(fp)[['Country Code', '2015 [YR2015]']].set_index('Country Code', drop=True).dropna()
        df.drop(df[df['2015 [YR2015]'] == '..'].index, inplace = True)
        df['2015 [YR2015]'] = np.array( df['2015 [YR2015]'], np.float32)
        self.hh_income_table = df

        # load gdl income map
        self.hh_income_map = MapReader(
            fp=os.path.join(self.data_folder, 'ECONOMY',
                            f'gdl_income_2015.tif'),
            xmin=self.model.xmin,
            ymin=self.model.ymin,
            xmax=self.model.xmax,
            ymax=self.model.ymax
        )

        # load scaled fixed migration costs
        fp = os.path.join(self.data_folder, 'ECONOMY', 'PROCESSED', 'scaled_fixed_migration_cost.csv')
        self.scaled_fixed_migration_cost = pd.read_csv(fp).set_index('iso3_code')

        # Get start date simulation and load 'closest' pop map
        start_sim = self.model.config['general']['start_time'].year
        GHSL_years = np.array([1990, 2000, 2015])
        year = GHSL_years[np.argmin(abs(GHSL_years - start_sim))]

        # load population map
        self.population = MapReader(
            fp=os.path.join(self.data_folder, 'POPULATION',
                            f'GHS_POP_{year}.tif'),
            xmin=self.model.xmin,
            ymin=self.model.ymin,
            xmax=self.model.xmax,
            ymax=self.model.ymax
        )

        # # Load ambient (natural) population change from Omphale 2013-2050
        self.nat_pop_change = pd.DataFrame([])

        # Load population change projections
        self.HistWorldPopChange = pd.read_excel(
            os.path.join(self.data_folder, 'POPULATION', 'WPP2019_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.xlsx'),
            sheet_name='ESTIMATES',
            skiprows=16)

        self.WorldPopChange = pd.read_excel(
            os.path.join(self.data_folder, 'POPULATION', 'WPP2019_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.xlsx'),
            sheet_name='MEDIUM VARIANT',
            skiprows=16)


        # Load urban settlemen layer
        self.SMOD = MapReader(
            fp=os.path.join(
                self.data_folder,
                'POPULATION',
                'SMOD',
                f'GHS_SMOD_E2015_GLOBE_R2022A_54009_1000_V1_0_WGS84.tif'),
            xmin=self.model.xmin,
            ymin=self.model.ymin,
            xmax=self.model.xmax,
            ymax=self.model.ymax
        )

        self.BUILT = MapReader(
            fp=os.path.join(
                self.data_folder,
                'POPULATION',
                f'GHS_BUILT_2015.tif'),
            xmin=self.model.xmin,
            ymin=self.model.ymin,
            xmax=self.model.xmax,
            ymax=self.model.ymax
        )


        # load PPP conversion rates
        path = os.path.join(self.data_folder, 'ECONOMY', 'conversion_rates_filled.csv')
        conversion_rates = pd.read_csv(path, index_col=0)
        self.conversion_rates = conversion_rates.set_index('alpha3', drop=True)

    def load_water_levels(self):
        # Load inundation maps to dicts
        self.inundation_maps_hist = {}
        self.inundation_maps_2030 = {}
        self.inundation_maps_2080 = {}
        rts = [1000, 500, 250, 100, 50, 25, 10, 5, 2]

        for i in rts:
            fp = os.path.join(
                self.data_folder,
                'SLR',
                'inundation_maps',
                f'inuncoast_historical_wtsub_hist_rp{(str(i).zfill(4))}_0.tif') # 
            self.inundation_maps_hist[i] = MapReader(
                fp=fp,
                xmin=self.model.xmin,
                ymin=self.model.ymin,
                xmax=self.model.xmax,
                ymax=self.model.ymax
            )

        if not self.model.args.rcp == 'control':
            for i in rts:
                # fill 2030
                fp = os.path.join(
                    self.data_folder,
                    'SLR',
                    'inundation_maps',
                    f'inuncoast_{self.model.args.rcp}_wtsub_2030_rp{(str(i).zfill(4))}_0.tif')
                    # f'merged_{self.model.args.rcp}_2030_rp{(str(i).zfill(4))}_flood_map.tif')

                self.inundation_maps_2030[i] = MapReader(
                fp=fp,
                xmin=self.model.xmin,
                ymin=self.model.ymin,
                xmax=self.model.xmax,
                ymax=self.model.ymax
            )

                # fill 2080
                fp = os.path.join(
                    self.data_folder,
                    'SLR',
                    'inundation_maps',
                    f'inuncoast_{self.model.args.rcp}_wtsub_2080_rp{(str(i).zfill(4))}_0.tif')
                    # f'merged_{self.model.args.rcp}_2080_rp{(str(i).zfill(4))}_flood_map.tif')

                self.inundation_maps_2080[i] = MapReader(
                    fp=fp,
                    xmin=self.model.xmin,
                    ymin=self.model.ymin,
                    xmax=self.model.xmax,
                    ymax=self.model.ymax
                )
