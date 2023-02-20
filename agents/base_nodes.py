import numpy as np
from numba import njit
import yaml
from honeybees.agents import AgentBaseClass
import geopandas as gpd
import os 
import pandas as pd

class HouseholdBaseClass(AgentBaseClass):
    """This class contains all the attributes, properties and functions that are shared by both the coastal and inland agent classes."""
    def __init__(self, model, agents):
        self.model = model
        self.agents = agents
                
        # Find admin
        self.geom_id = self.geom['properties']['id'] 
        
        # Sample region income and distribution for both coastal and inland
        data = self.model.data.hh_income.sample_geom(self.geom)
        data = data.ravel()
        data = data[data != -1]
        income_region = np.median(data) 
        self.income_region = income_region 

        # Sample average age coastal and inland
        data = self.model.data.mean_age.sample_geom(self.geom)
        data = data.ravel()
        data = data[data != -1]
        average_age_node = np.median(data) 
        self.average_age_node = average_age_node 

        # Create income distribution for each region (n=5_000)
        mean_income = income_region * self.model.settings['adaptation']['mean_median_inc_ratio']
        mu = np.log(income_region)
        sd = np.sqrt(2*np.log(mean_income/income_region))
        self.income_distribution_region = np.sort(np.random.lognormal(mu, sd, 5_000).astype(np.int32)) # initiate with 2_000 
        self.average_household_income = int(self.income_distribution_region.mean())

        # Initiate the percentage of households implementing dry proofing for all regions
        self.percentage_adapted = None
        self.n_households_adapted = None
        self.perc_people_moved_out = 0
        self.flood_tracker = 0
        self.total_shoreline_change_admin  = 0
        self.nearshore_slopes_admin = None
        self.segment_IDs_admin = None
        self.average_beach_width = 0
        self.beach_length = 0
        self.summed_beach_amenity = 0
        self.beach_amenity_dict = {}
        # Initiate expected damages for all regions
        self.ead_total = 0
        self.people_near_beach = 0
        self.households_near_beach = 0

        self.n_moved_out_last_timestep = 0
        self.n_moved_in_last_timestep = 0
        self.people_moved_out_last_timestep = 0

        self.read_location_effect_gravity_model()
        self.initiate_agents()
 


        # Extract gadm name of floodplain and corresponding inland regions (should be moved to initiate agents)
        coastal_admins = [region['properties']['id'] for region in self.model.area.geoms['admin'] if region['properties']['id'].endswith('flood_plain')]
        
        # List comprehension crashes
        for index in range(len(coastal_admins)):
            coastal_admins.append(coastal_admins[index][:-12])
        self.coastal_admins = coastal_admins


        self.average_amenity_value = 0 # Average amenity value in all regions (coastal will be overwritten). Improve this

    def read_location_effect_gravity_model(self):
        
        if self.model.settings['gravity_model']['inlcude_location_effects']:
            raise NotImplementedError()
            # load gadm shape
            gadm_shape = gpd.read_file(os.path.join('DataDrive', 'SLR', 'GADM', 'GADM_2.shp')) 
            gadm_translate = pd.DataFrame({'admin_name': list(gadm_shape[f'NAME_{self.model.args.admin_level}'])}, index=gadm_shape[f'GID_{self.model.args.admin_level}'])
            
            if self.geom_id.endswith('flood_plain'):
                admin_name = gadm_translate.loc[self.geom_id[:-12]]['admin_name']
                # admin_name = self.geom_id[:-12]

            else:
                admin_name = gadm_translate.loc[self.geom_id]['admin_name']
                # admin_name = self.geom_id


            # read fixed destination effect from table 
            if admin_name in self.model.data.location_effects.index:
                self.origin_effect_gravity_model = self.model.data.location_effects.loc[admin_name]['origin_effect']
                self.destination_effect_gravity_model = self.model.data.location_effects.loc[admin_name]['destination_effect']
            else:
                print(admin_name)
                self.origin_effect_gravity_model = 0
                self.destination_effect_gravity_model = 0

        else:
            self.origin_effect_gravity_model = 0
            self.destination_effect_gravity_model = 0


    @staticmethod
    @njit(cache=True)
    def return_household_sizes(flow_people, max_household_size):
        '''Small helper function to speed up sampling households from people flow'''
        # Preallocate empty array
        household_sizes = np.full(flow_people, -1, dtype=np.int16) # Way too big   
        i = 0
        while flow_people > max_household_size:
            household_size = min([flow_people, np.random.randint(1, max_household_size)])
            flow_people -= household_size
            household_sizes[i] = household_size
            i += 1
            # allocate last household
        household_sizes[i] = flow_people
        household_sizes = household_sizes[np.where(household_sizes != -1)]
        return household_sizes   

    def ambient_pop_change(self):
        # Process population growth
        population_change =  self.agents.population_data.loc[self.geom_id]['change']
        international_migration_pp = self.model.settings['gravity_model']['annual_international_migration']/ np.sum(self.agents.regions.population)
        population_change += np.floor(international_migration_pp * self.population)


        # No nat pop change in spin up period
        if self.model.config['general']['start_time'].year == self.model.current_time.year:
            population_change = 0

        # Generate households from new people
        household_sizes = self.return_household_sizes(int(abs(population_change)), self.max_household_size)
        return population_change, household_sizes     

    # def gini_coefficient(self, array):
    #     """Calculate the Gini coefficient of a numpy array."""
    #     # Source dir: https://github.com/oliviaguest/gini 
    #     # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    #     # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    #     gini_array = array.copy() #all values are treated equally, arrays must be 1d
    #     gini_array = gini_array*1E-3
    #     gini_array = np.sort(gini_array) #values must be sorted
    #     index = np.arange(1,gini_array.shape[0]+1) #index per array element
    #     n = gini_array.shape[0]#number of array elements
    #     return ((np.sum((2 * index - n  - 1) * gini_array)) / (n * np.sum(gini_array))) #Gini coefficient

    # Generate households moving out of inland nodes
    @staticmethod
    @njit
    def _generate_households(
        n_households_to_move,
        household_sizes, 
        move_to_region_per_household,
        hh_risk_aversion, init_risk_perception, 
        income_percentiles,
        ):
        # Generate people moving out
        # sum household sizes to find total number of people moving
        n_movers = int(household_sizes.sum())
        # create output arrays people
        to_region = np.full(n_movers, -1, dtype=np.int32)
        household_id = np.full(n_movers, -1, dtype=np.int32)
        gender = np.full(n_movers, -1, dtype=np.int8)
        risk_aversion = np.full(n_movers, -1, dtype=np.float32)
        age = np.full(n_movers, -1, dtype=np.int8)
        income_percentile = np.full(n_households_to_move, -99, dtype=np.int32)
        income = np.full(n_movers, 0, dtype=np.float32)
        risk_perception = np.full(n_movers, init_risk_perception, dtype=np.float32)

        # fill households
        start_idx = 0
        for i in range(n_households_to_move):
            end_idx = start_idx + int(household_sizes[i])
            to_region[start_idx: end_idx] = move_to_region_per_household[i]
            household_id[start_idx: end_idx] = i
            gender[start_idx: end_idx] = np.random.randint(0, 2, size=end_idx - start_idx)
            risk_aversion[start_idx: end_idx] = np.full((end_idx - start_idx), hh_risk_aversion) 
            age[start_idx: end_idx] = np.random.randint(0, 85, size=end_idx - start_idx)

            start_idx = end_idx

        assert end_idx == n_movers
        return n_movers, to_region, household_id, gender, risk_aversion, age, income_percentile, income, risk_perception