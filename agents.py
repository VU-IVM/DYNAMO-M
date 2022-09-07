import numpy as np
from numba import njit
import numba as nb
import yaml
import random
import os
import pandas as pd
import geopandas as gpd
from honeybees.agents import AgentBaseClass
from honeybees.library.raster import pixels_to_coords, pixel_to_coord
from honeybees.library.neighbors import find_neighbors
from scipy.spatial import distance_matrix as sdistance_matrix
from scipy import interpolate
from decision_module import calcEU_no_nothing, calcEU_adapt, EU_migrate, gravity_model

from flood_risk_module import stochastic_flood
from export_agents import export_agents, export_matrix, export_agent_array
from node_properties import NodeProperties, CoastalNodeProperties, InlandNodeProperties
from population_change import WorldPopProspectsChange

class HouseholdBaseClass(AgentBaseClass):
    def __init__(self, model, agents):
    # Load agent settings
        with open(model.settings_path) as f:
            self.settings = yaml.load(f, Loader=yaml.FullLoader)
        self.model = model
        self.agents = agents
        
        # Find admin name 
        self.admin_name = self.geom['properties']['id']
        
        # Sample region income and distribution for both coastal and inland
        data = self.model.data.hh_income.sample_geom(self.geom)
        data = data.ravel()
        data = data[data != -1]
        income_region = np.median(data) 
        self.income_region = income_region 

        # Create income distribution for each region (n=5_000)
        mean_income = income_region * self.settings['adaptation']['mean_median_inc_ratio']
        mu = np.log(income_region)
        sd = np.sqrt(2*np.log(mean_income/income_region))
        self.income_distribution_region = np.sort(np.random.lognormal(mu, sd, 5_000).astype(np.int32)) # initiate with 2_000 
    
        # Sample unemployment rates for both coastal and inland
        data = self.model.data.unemployment_rate.sample_geom(self.geom)
        data = data.ravel()
        data = data[data != -1]
        unemployment_rate = np.round(np.median(data)) 
        if np.isnan(unemployment_rate):
            unemployment_rate = 0
        self.unemployment_rate = unemployment_rate

        # Initiate the percentage of households implementing dry proofing for all regions
        self.percentage_adapted = None
        self.n_people_adapted = None
        self.perc_people_moved_out = 0
        self.n_people_adapted = 0
        self.flood_tracker = 0

        # Initiate expected damages for all regions
        self.ead_total = 0

        self.n_moved_out_last_timestep = 0
        self.n_moved_in_last_timestep = 0
        self.people_moved_out_last_timestep = 0

        self.initiate_agents()

        self.average_amenity_value = 0 # Average amenity value in all regions (coastal will be overwritten). Improve this

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
        population_change =  self.agents.population_data.loc[self.admin_name]['change']


        international_migration_pp = self.settings['gravity_model']['annual_international_migration']/ np.sum(self.agents.regions.population)
        population_change += np.floor(international_migration_pp * self.population)


        # No nat pop change in spin up period
        if self.model.config['general']['start_time'].year == self.model.current_time.year:
            population_change = 0

        # Generate households from new people
        household_sizes = self.return_household_sizes(int(abs(population_change)), self.max_household_size)
        return population_change, household_sizes     

    def update_income_distribution_region(self):          
        self.income_region = round(np.median(self.income))
 
    # Generate households moving out of inland nodes
    @staticmethod
    @njit
    def _generate_households(n_households_to_move, household_sizes, move_to_region_per_household,
                             hh_risk_aversion, init_risk_perception, income_percentiles):
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


    @property
    def n_moved_out_last_timestep(self):
        return self._n_moved_out_last_timestep

    @n_moved_out_last_timestep.setter
    def n_moved_out_last_timestep (self, value):
        self._n_moved_out_last_timestep = value

    @property
    def n_moved_in_last_timestep(self):
        return self._n_moved_in_last_timestep

    @n_moved_in_last_timestep.setter
    def n_moved_in_last_timestep(self, value):
        self._n_moved_in_last_timestep = value


class CoastalNode(HouseholdBaseClass, CoastalNodeProperties):
    """General household class.
    
    This class contains households, and the people within these households. The household attributes, such as their size are held in arrays with size n.::

        self.size = [3, 2, 3, 2, .., 4, n]

    Other "per-person"-arrays can contain information about agents themselves, such as their age.::

        self.age = [58, 62, 26, 40, 40, 81, 80]

    In addition, `self.people_indices_per_household` maps the people in these households to positions in the per-person arrays.::

        self.people_indices_per_household = [
            [0, 1, 2, -1, -1],
            [5, 6, -1, -1, -1]
            [3, 4, -1, -1, -1],
            ...,
            [.., .., .., .., ..]
        ]

    In the example above each household has a maximum size of 5. The first household is represented by the first row. Here it shows that the first household is of size 3 (matches first item in `self.size`). The other "spots" in the household are empty as represented by -1. As the first household is made up by the first, second and third indices, the respective ages of the people in that household are 58, 62 and 62.

    The people in the second household are 81 and 80 (as represented by the 5 and 6) and the people in the third household are both 40.

    Another array `self.household_id_per_person` which contains the household id for each of the agent, which is basically the inverse of `self.people_indices_per_household`.::

        self.household_id_per_person = [0, 0, 0, 2, 2, 1, 1, .., ..]

    This indicates that the first three persons are in the first (1st household, the next 2 persons are in the 3th household and the next 2 persons are in the 2nd household).

    Args:
        model: The model class.
        agents: The agents class.
        geom: The geometry of the region.
        distance_matrix: Matrix of distances to other regions.
        n_households_per_region: Vector of the number of households per region.
        idx: Index of the current region.
        reduncancy: Number of empty spaces for new households in region (model will crash if number of households becomes higher than redundancy)
        person_reduncancy: Number of empty spaces for new persons in region (model will crash if number of households becomes higher than redundancy).
        init_folder: Folder with initalization files.
        max_household_size: The maximum size of a household in this region.
    """
    def __init__(self, model, agents, geom: dict, distance_matrix: np.ndarray, n_households_per_region: np.ndarray, idx: int, redundancy: int, person_reduncancy: int, init_folder: str, max_household_size: int):
        self.redundancy = redundancy
        self.person_reduncancy = person_reduncancy
        self.geom = geom
        self.admin_idx = idx
        self.distance_vector = distance_matrix[idx]
        self.n_households_per_region = n_households_per_region
        self.init_folder = init_folder
        self.max_household_size = max_household_size
       
        super(CoastalNode, self).__init__(model, agents) 
        HouseholdBaseClass.__init__(self, model, agents)

        # amenity_values_region = self.model.data.amenity_value.get_array()
        # amenity_values_region[amenity_values_region<0] = 100 # PRELIMINARY WAY TO DEAL WITH NODATA VALUES
        # self.amenity_values_region = amenity_values_region

    
    def initiate_agents(self):
        self._initiate_locations()
        self._initiate_household_attributes()
        self._initiate_person_attributes()

    def _load_initial_state(self):
        self.load_timestep_data()

    def _initiate_locations(self) -> None:
        """Loads household locations from file. Also sets the number of households (`self.n`) and the maximum number of households (`self.max_n`) using the redundancy paramter."""
        locations_fn = os.path.join(self.init_folder, "locations.npy")
        if os.path.exists(locations_fn):
            household_locations = np.load(locations_fn)
        else:
            household_locations = np.zeros((0, 2), dtype=np.float32)

        self.n = household_locations.shape[0]
        self.max_n = self.n + self.redundancy
        assert self.max_n < 4294967295 # max value of uint32, consider replacing with uint64
        
        self._locations = np.full((self.max_n, 2), np.nan, dtype=np.float32)
        self.locations = household_locations

    @staticmethod
    @njit
    def _generate_household_id_per_person(people_indices_per_household: np.ndarray, n_people: int) -> np.ndarray:
        """Generates an array that can be used to find the household index for each person.
        
        This array contains the household id for each agent, which is basically the inverse of `self.people_indices_per_household`.::

            self.household_id_per_person = [0, 0, 0, 2, 2, 1, 1, .., ..]

        This indicates that the first three persons are in the first (1st household, the next 2 persons are in the 3th household and the next 2 persons are in the 2nd household).

        Args:
            people_indices_per_household: An array where the first dimension represents the different households, and the second dimension the people in those households.
            n_people: The current number of people living in the region.

        Returns:
            household_id_per_person: Array containing the household id for each agent.
        """
        n_people = np.count_nonzero(people_indices_per_household != -1)
        household_id_per_person = np.full(n_people, -1, dtype=np.int32)
        n_households, max_household_size = people_indices_per_household.shape
        
        k = 0
        for household_n in range(n_households):
            for j in range(max_household_size):
                if people_indices_per_household[household_n, j] == -1:
                    break
                household_id_per_person[k] = household_n
                k += 1

        assert (household_id_per_person != -1).all()
        return household_id_per_person
    
    def _initiate_household_attributes(self):
        self._size = np.full(self.max_n, -1, dtype=np.int32)
        size_fn = os.path.join(self.init_folder, "size.npy")
        if os.path.exists(size_fn):
            self.size = np.load(size_fn)

        self.population = np.sum(self.size) # calculate population size in coastal node

        self._people_indices_per_household = np.full((self.max_n, self.max_household_size), -1, dtype=np.int32)
        people_indices_per_household_fn = os.path.join(self.init_folder, "people_indices.npy")
        if os.path.exists(people_indices_per_household_fn):
            self.people_indices_per_household = np.load(people_indices_per_household_fn)

        # Initiate wealth, income, flood experience and adaptation status
        self._ead = np.full(self.max_n, -1)
        self._ead_dryproof = np.full(self.max_n, -1)
        self._adapt = np.full(self.max_n, -1)
        self._time_adapt = np.full(self.max_n, -1)
        self._income = np.full(self.max_n, -1)
        self._flooded = np.full(self.max_n, -1)
        self._flood_count = np.full(self.max_n, -1)
        self._amenity_value = np.full(self.max_n, -1)


        self.flooded = 0
        self.flood_count = 0

        # Position in the income distribution (related to eduction/ age etc)
        self._income_percentile = np.full(self.max_n, -1)
        self.income_percentile = np.random.randint(0, 100, self.n)

        self._income = np.full(self.max_n, -1)
        self.income = np.percentile(self.income_distribution_region, self.income_percentile)
                          
        self._property_value = np.full(self.max_n, -1)
        self.property_value = self.settings['flood_risk_calculations']['property_value']

        # Create dict of income/ wealth ratio and interpolate
        perc = np.array([0, 20, 40, 60, 80, 100])
        ratio = np.array([0, 1.06, 4.14, 4.19, 5.24, 6])
        self.income_wealth_ratio = interpolate.interp1d(perc, ratio)

        self._wealth =  np.full(self.max_n, -1)
        self.wealth =  self.income_wealth_ratio(self.income_percentile) * self.income
        self.wealth[self.wealth < self.property_value] = self.property_value[self.wealth < self.property_value] 
        
        self._decision_horizon = np.full(self.max_n, -1)
        self.decision_horizon = self.settings['decisions']['decision_horizon']
        
        self._hh_risk_aversion = np.full(self.max_n, -1, dtype= np.float32)
        self.hh_risk_aversion = self.settings['decisions']['risk_aversion']

        self._risk_perception = np.full(self.max_n, -1, dtype = np.float32)
        self.risk_perception = self.settings['flood_risk_calculations']['risk_perception']['min']

        self._flood_timer = np.full(self.max_n, -1, dtype = np.int32)
        self.flood_timer = 99 # Assure new households have min risk perceptions

        # Initiate amenity distance decay function (interpolate between values of Conroy & Milosch 2011)
        dist_km = np.array([0, .500, 1, 10, 20, 1E6])
        amenity_premium = np.array([0.60, 0.60, 0.10, 0.03, 0, 0])

        self.coastal_amenity_function =  interpolate.interp1d(dist_km, amenity_premium)

        dist_to_coast = np.ceil(self.model.data.dist_to_coast.sample_coords(self.locations, cache=True) * 2)/ 2
        dist_to_coast[dist_to_coast == 0] = 0.5
        self.amenity_value = self.coastal_amenity_function(dist_to_coast) * self.wealth 
        
        # self.amenity_value = 10_000
        self.average_amenity_value = 0

    def sample_water_level(self):
        """This function creates a dictionary of water levels for inundation events of different return periods.
        It uses the sample coordenates method of the ArrayReader class instances loaded in data.py. The inundation maps
        are selected based on the scenario defined in the terminal command 'rcp'."""
        
        self.rts = np.array([1000, 500, 250, 100, 50, 25, 10, 5, 2])
        self.water_level_hist = {}
        self.water_level_2080 = {}

        # Fill water levels by sampling agent locations
        
        for i in self.rts:
            self.water_level_hist[i] =  self.model.data.inundation_maps_hist[i].sample_coords(self.locations, cache=True)

        if self.model.args.rcp == 'control':
            self.water_level_2080 = self.water_level_hist
        else:
            for i in self.rts:
                self.water_level_2080[i] =  self.model.data.inundation_maps_2080[i].sample_coords(self.locations, cache=True)
        
        # Interpolate water level between year 2000 and 2080
        self.water_level = {}
    
        # Extract start time from config
        start_time = self.model.config['general']['start_time'].year
        timestep = self.model.current_time.year - start_time

        # Derive current water depth based on linear interpolation
        for rt in self.rts:
            self.water_level_hist[rt][self.water_level_hist[rt] < 0.001] = 0
            self.water_level_2080[rt][self.water_level_2080[rt] < 0.001] = 0
            difference = (self.water_level_2080[rt] - self.water_level_hist[rt])/ (2080 - start_time)
            self.water_level[rt] = self.water_level_hist[rt] + difference * timestep
        
        # Read protection standard and set inundations levels to 0 if protected
        fps = self.settings['flood_risk_calculations']['flood_protection_standard']
        for rt in self.water_level:
            if rt < fps:
                self.water_level[rt] = np.zeros(len( self.water_level[rt]))

    def calculate_ead(self):
        # Interpolate damage factor based on damage curves  
        func_dam = interpolate.interp1d(self.model.data.curves['index'],self.model.data.curves[0])
        func_dam_dryproof_1m = interpolate.interp1d(self.model.data.curves_dryproof_1m['index'],self.model.data.curves_dryproof_1m[0])
        
        # Indicate maximum damage per household
        max_dam = self.property_value
        # pre-allocate empty array with shape (3, self.n) for number of damage levels and number of households
        self.damages = np.zeros((len(self.rts), self.n), dtype=np.float32)
        
        for i, rt in enumerate(self.rts):
            self.water_level[rt][self.water_level[rt] < 0] = 0
            self.water_level[rt][self.water_level[rt] > 6] = 6
            # calculate damage per retun period and store in damage dictory
            # place the damage output in the empty array
            self.damages[i] = func_dam(self.water_level[rt]) * max_dam
        x = 1 / self.rts
        # calculate ead on damage array along the first axis
        self.ead = np.trapz(self.damages, x, axis=0)        
        
        # pre-allocate empty array with shape (3, self.n) for number of damage levels and number of households
        self.damages_dryproof_1m = np.zeros((len(self.rts), self.n), dtype=np.float32)
        for i, rt in enumerate(self.rts):
            self.water_level[rt][self.water_level[rt] < 0] = 0
            self.water_level[rt][self.water_level[rt] > 6] = 6
            # calculate damage per retun period and store in damage dictory
            # place the damage output in the empty array
            self.damages_dryproof_1m[i] = func_dam_dryproof_1m(self.water_level[rt]) * max_dam
        x = 1 / self.rts
        # calculate ead on damage array along the first axis
        self.ead_dryproof = np.trapz(self.damages_dryproof_1m, x, axis=0)  
        
        # Sum and update expected damages per node
        agents_that_adapted = np.where(self.adapt == 1)
        agents_not_adapted = np.where(self.adapt == 0)
        
        self.ead_total = np.sum(self.ead[agents_not_adapted]) + np.sum(self.ead_dryproof[agents_that_adapted])

    def _initiate_person_attributes(self):
        n_people = np.count_nonzero(self._people_indices_per_household != -1)
               
        self._empty_index_stack = np.full(n_people + self.person_reduncancy, -1, dtype=np.int32)
        size_empty_stack = self._empty_index_stack.size-n_people
        self._empty_index_stack[:size_empty_stack] = np.arange(n_people, self._empty_index_stack.size)[::-1]
        self._empty_index_stack_counter = size_empty_stack - 1

        self._gender = np.full(self._empty_index_stack.size, -1, dtype=np.int8)
        gender_fn = os.path.join(self.init_folder, "gender.npy")
        if os.path.exists(gender_fn):
            self._gender[:n_people] = np.load(gender_fn)
            
        self._age= np.full(self._empty_index_stack.size, -1, dtype=np.int8)
        age_fn = os.path.join(self.init_folder, "age.npy")
        if os.path.exists(age_fn):
            self._age[:n_people] = np.load(age_fn)

        self._household_id_per_person = np.full(self.max_n_people, -1, dtype=np.int32)
        self.household_id_per_person = self._generate_household_id_per_person(self._people_indices_per_household, self.size.sum())
        
        hh_risk_aversion = np.full(self.n, self.settings['decisions']['risk_aversion'], dtype=np.float32)
        person_risk_aversion = np.take(hh_risk_aversion, self.household_id_per_person)     
        
        # # create redundancy array for concentenating 
        redundancy_array = np.full(self.person_reduncancy, -1, dtype=np.float32)  
        
        #concenate and store result
        self._risk_aversion = np.concatenate((person_risk_aversion, redundancy_array), axis = 0)

        #Assert shape
        assert self._gender.shape == self._risk_aversion.shape
        assert self._age.shape == self._gender.shape 
        
        #Assert data
        assert np.array_equal(self._gender == -1, self._risk_aversion == -1)
        assert np.array_equal(self._gender == -1, self._age == -1)
        
        return None

    def load_timestep_data(self):
        pass

    def process(self):
        self._household_id_per_person = self._generate_household_id_per_person(self._people_indices_per_household, self.size.sum())       

    def initiate_household_attributes_movers(self, n_movers):
        '''This function assigns new household attributes to the households that moved in. It takes all arrays and fills in the missing data based on sampling.'''
        
        assert self.income[-n_movers] == -1 
        
        # Sample income percentile for households moving in from inland node or natural pop change
        # Find neighbors for newly generated households
        new_households = np.where(self.income_percentile == -99)[0]   

        # Fill
        neighbor_households_1km = find_neighbors(locations=self.locations, radius=1_000, n_neighbor=30, bits=32, search_ids=new_households)
        neighbor_households_3km = find_neighbors(locations=self.locations, radius=3_000, n_neighbor=30, bits=32, search_ids=new_households)


        for i, household in enumerate(new_households):
            # Get neighbors
            neighboring_hh = neighbor_households_1km[i, :]
            
            if neighboring_hh.size == 0:             

                # If no neighbors found increase search radius
                neighboring_hh = neighbor_households_3km[i, :]
                
                if neighboring_hh.size == 0:
                    print('No neighbors found')
                    

            # Filter out no data
            neighboring_hh = neighboring_hh[neighboring_hh < self.n]

            # Select income percentile neighbors
            income_percentile_neighbors = self.income_percentile[neighboring_hh]
            
            # Filter out new households if selected as neighbor
            income_percentile_neighbors = income_percentile_neighbors[income_percentile_neighbors!=-99]
            
            # Sample income percentile from neighbors
            if income_percentile_neighbors.size != 0:
                self.income_percentile[household] = np.random.choice(income_percentile_neighbors)
            else:
                self.income_percentile[household] = np.random.choice(self.income_percentile[self.income_percentile != -99])

        # Generate wealth and income based on agent position in the income distribution
        self.income[-n_movers:self.n] = np.percentile(self.income_distribution_region, self.income_percentile[-n_movers:self.n])
        assert (self.income > 0).all()
        
        self.wealth[-n_movers:self.n]= self.income[-n_movers:self.n] * self.income_wealth_ratio(self.income_percentile[-n_movers:self.n])
        
        # All agents have the same property value
        self.property_value[-n_movers:self.n] = self.settings['flood_risk_calculations']['property_value']

        # Set wealth to never be lower than property value
        self.wealth[self.wealth < self.property_value] = self.property_value[self.wealth < self.property_value] 
        
        # Set decision horizon and flood timer
        self.decision_horizon[self.decision_horizon == -1] = self.settings['decisions']['decision_horizon']
        self.flood_timer[self.flood_timer == -1] = 99

        # Initiate amenity distance decay function (interpolate between values of Conroy & Milosch 2011)
        dist_to_coast = np.ceil(self.model.data.dist_to_coast.sample_coords(self.locations, cache=True) * 2)/ 2
        dist_to_coast[dist_to_coast == 0] = 0.5
        self.amenity_value = self.coastal_amenity_function(dist_to_coast) * self.wealth 
        
        # self.amenity_value = 10_000
        self.average_amenity_value = 0

        # Reset flood status to 0 for all households (old and new) and adaptation to 0 for new households
        self.adapt[-n_movers:] = 0
        self.flood_count[-n_movers:] = 0
        
        # Uncomment if we want to update income based on agent population in the flood zone
        # self.update_income_distribution_region() 

    def update_flood(self):
        '''In this function the flood risk perception attribute of each agent is updated through simulated flood events using the
        flood risk module.
        '''
        # reset flooded to 0 for all households
        self.flooded = 0 
        self.flood_tracker = 0
        self.flooded, self.flood_count, self.risk_perception, self.flood_timer, self.flood_tracker = stochastic_flood(
            water_levels= self.water_level,
            return_periods=self.rts,
            flooded=self.flooded,
            flood_count=self.flood_count,
            risk_perceptions=self.risk_perception,
            flood_timer = self.flood_timer,
            risk_perc_min = self.settings['flood_risk_calculations']['risk_perception']['min'],
            risk_perc_max = self.settings['flood_risk_calculations']['risk_perception']['max'],
            risk_decr = self.settings['flood_risk_calculations']['risk_perception']['coef'],
            settings = self.settings['general']['flood'],
            current_year=self.model.current_time.year,
            spin_up_flag = self.model.spin_up_flag,
            flood_tracker= self.flood_tracker)       

    def process_population_change(self):
        population_change, household_sizes = self.ambient_pop_change()
        # Select households to remove from
        households_to_remove = []
        if population_change < 0 and self.n > household_sizes.size:
            for size in np.sort(household_sizes):
                # find a corresponding household size and remove (if not found remove smallest household for now)
                try:
                    household_to_remove = np.where(self.size == size)[0][0]
                except:
                    household_to_remove = np.where(self.size == np.sort(self.size)[0])[0][0]
                # Check if household is already tagged for removal
                j = 0
                while household_to_remove in households_to_remove:
                    household_to_remove = np.where(self.size == size)[0][j]
                    j += 1
                households_to_remove.append(household_to_remove)

            households_to_remove = np.sort(households_to_remove)[::-1]

            n_movers = np.sum(household_sizes)
            move_to_region = np.full(n_movers, self.admin_idx, dtype=np.int32) # Placeholder, will not do anythin with this 

            # Remove households from abm
            self.population, self.n, self._empty_index_stack_counter, _, _, _, _, _, _, _, _, _ = self.move_numba(
                        self.population,
                        self.n,
                        self._people_indices_per_household,
                        self._empty_index_stack,
                        self._empty_index_stack_counter,
                        households_to_remove,
                        n_movers,
                        move_to_region,
                        self.admin_idx,
                        self._locations,
                        self._size,
                        self._ead,
                        self._ead_dryproof,
                        self._gender,
                        self._risk_aversion,
                        self._age,
                        self._income_percentile,
                        self._income,
                        self._wealth,
                        self._risk_perception,
                        self._flood_timer,
                        self._adapt,
                        self._time_adapt,
                        self._decision_horizon,
                        self._hh_risk_aversion,
                        self._property_value,
                        self._amenity_value
                    )

        elif population_change > 0:
            # Generate attributes
            n_movers, to_region, household_id, gender, risk_aversion, age, income_percentile, income, risk_perception = self._generate_households(
                n_households_to_move=household_sizes.size,
                household_sizes=household_sizes,
                move_to_region_per_household = np.full(household_sizes.size, self.admin_idx), # new households as 'moved' to own region
                hh_risk_aversion = self.settings['decisions']['risk_aversion'],
                init_risk_perception = self.settings['flood_risk_calculations']['risk_perception']['min'],
                income_percentiles = nb.typed.List(self.model.agents.regions.income_percentiles_regions)
            )
            person_income_percentile = np.take(income_percentile, household_id)
            person_risk_perception =  np.take(risk_perception, household_id)
            person_income = np.take(income, household_id)

            people = {"from": np.full(n_movers, self.admin_idx, dtype=np.int32),
            "to": to_region,
            "household_id": household_id,
            "gender": gender,
            "risk_aversion": risk_aversion,
            "age": age,
            "income_percentile": person_income_percentile,
            "income": person_income,
            "risk_perception": person_risk_perception,
            }

            # Add households
            self.add(people)
        
        elif population_change == 0:
            pass   

           

    def step(self):
        if self.model.current_time.year > self.model.config['general']['start_time'].year:
            self.process_population_change()
        self.load_timestep_data()
        self.process()
        self.sample_water_level()
        self.calculate_ead()
        self.update_flood()

    @staticmethod
    @njit
    def add_numba(
        n: int,
        people_indices_per_household: np.ndarray,
        household_id_per_person: np.ndarray,
        empty_index_stack: np.ndarray,
        empty_index_stack_counter: int,
        index_first_persons_in_household: np.ndarray,
        new_household_sizes: np.ndarray,
        new_income_percentiles: np.ndarray,
        new_risk_perceptions: np.ndarray,
        new_risk_aversions: np.ndarray,
        gender_movers: np.ndarray,
        age_movers: np.ndarray,
        risk_aversion_movers: np.ndarray,
        locations: np.ndarray,
        size: np.ndarray,
        gender: np.ndarray,
        risk_aversion: np.ndarray,
        age: np.ndarray,
        income_percentile: np.ndarray,
        risk_perception: np.ndarray,
        hh_risk_aversion: np.ndarray,
        admin_indices: np.ndarray,
        p_suitability: np.ndarray, 
        gt: tuple[float, float, float, float, float, float],
    ) -> None:
        """This function adds new households to a region. As input the function takes as input the characteristics of the people moving in, and inserts them into the current region. For example, for individual people the data from `age_movers` is inserted into `age`. Likewise the size of the household is inserted into `size`.
        
        Args:
            n: Current number of households in the region.
            people_indices_per_household: maps the people in these households to positions in the per-person arrays.
            household_id_per_person: Household id for each person. (TODO: why is this not used?)
            empty_index_stack: An array of indices with empty household ids, for example when a household moved.
            empty_index_stack_counter: The current stack index for `empty_index_stack`.
            index_first_persons_in_household: Array of index of the first person in each of the incoming households. For example, when 10 people from 2 households are moving in, and the first houshold consist of 6 people, while the second household consists of the other 4 people. This array should contain [0, 5]; 0 for the index of the first houshold and 5 for the first index of the second household.
            new_household_sizes: The size of each of the new households.
            gender_movers: The gender of each of the people moving in.
            age_movers: The age of each of the people moving in.
            risk_aversion_movers: The risk of each of the people moving in.
            locations: The array that contains the locations of all households in the destination (the households moving in are inserted here).
            size: The array that contains the size of all households in the destination (the households moving in are inserted here).
            gender: The array that contains the gender of all people in the destination (the people moving in are inserted here).
            risk_aversion: The array that contains the risk aversion of all people in the destination (the people moving in are inserted here).
            age: The array that contains the age of all people in the destination (the people moving in are inserted here).
            admin_indices: The cell indices of the admin units; used to determine the coordinates of that people are moving to. Currently selected at random.
            gt: The geotransformation for the cell indices.
        """

        all_cells = np.arange(0, admin_indices[0].size, dtype=np.float32)       

        for index_first_person_in_household, new_household_size, new_income_percentile, new_risk_perception, new_risk_aversion in zip(index_first_persons_in_household, new_household_sizes, new_income_percentiles, new_risk_perceptions, new_risk_aversions):
            n += 1
            assert size[n-1] == -1
            assert income_percentile[n-1] == -1

            size[n-1] = new_household_size
            income_percentile[n-1] = new_income_percentile
            risk_perception[n-1] = new_risk_perception
            hh_risk_aversion[n-1] = new_risk_aversion

            # Fill individual attributes
            for i in range(new_household_size):
                # get an empty index
                empty_index = empty_index_stack[empty_index_stack_counter]
                # check if we don't get an empty index
                assert empty_index != -1
                # check if spot is empty
                assert gender[empty_index] == -1
                assert risk_aversion[empty_index] == -1
                assert age[empty_index] == -1
                # set gender, risk aversion and age
                gender[empty_index] = gender_movers[index_first_person_in_household + i]
                risk_aversion[empty_index] = risk_aversion_movers[index_first_person_in_household + i]
                age[empty_index] = age_movers[index_first_person_in_household + i]
                # emtpy the index stack
                empty_index_stack[empty_index_stack_counter] = -1
                # and finally decrement stack counter
                empty_index_stack_counter -= 1
                if empty_index_stack_counter < 0:
                    raise OverflowError("Too many agents in class. Consider increasing redundancy")
                assert empty_index_stack[empty_index_stack_counter] != -1
                # set empty index in people indices
                people_indices_per_household[n-1, i] = empty_index
            
            

            # Uncomment for random allocation in the flood zone
            # cell = random.randint(0, admin_indices[0].size-1) # Select random cell

            # Select random cell based on probabilities derived from suitability values
            cell = int(all_cells[np.searchsorted(np.cumsum(p_suitability), np.random.rand())])

            # Allocate randomly in 1km2 grid cell (otherwise all agents the center of the cell)
            px = admin_indices[1][cell] + random.random()
            py = admin_indices[0][cell] + random.random()

            locations[n-1] = pixel_to_coord(px, py, gt)

        return n, empty_index_stack_counter
    
    def add(self, people: dict[np.ndarray]) -> None:
        """This function adds new households to a region. As input the function takes a dictionary of people characteristics, such as age and gender. In addition, the key and corresponding array 'household_id' is used to determine the household that the person belongs to."""
        index_first_persons_in_household, new_household_sizes = np.unique(people['household_id'], return_index=True, return_counts=True)[1:]
        
        # Extract household income percentile from people array
        new_income_percentiles = people['income_percentile'][index_first_persons_in_household]
        new_risk_perceptions =  people['risk_perception'][index_first_persons_in_household]
        new_risk_aversions = people['risk_aversion'][index_first_persons_in_household]

        self.n_moved_in_last_timestep = index_first_persons_in_household.size
        
        # assign suitability matrix
        suitability = self.model.data.suitability_arr 
        admin_indices = self.geom['properties']['gadm']['indices']
        
        if np.sum(suitability[admin_indices]) == 0:
           suitability[admin_indices] = 0.01 # If no urban area present distribute agents randomnly within flood zone
           print(f'no urban area in {self.admin_name}')

        p_suitability = suitability[admin_indices]/np.sum(suitability[admin_indices])

        self.n, self._empty_index_stack_counter = self.add_numba(
            n=self.n,
            people_indices_per_household=self._people_indices_per_household,
            household_id_per_person=self._household_id_per_person,
            empty_index_stack=self._empty_index_stack,
            empty_index_stack_counter=self._empty_index_stack_counter,
            index_first_persons_in_household=index_first_persons_in_household,
            new_household_sizes=new_household_sizes,
            new_income_percentiles=new_income_percentiles,
            new_risk_perceptions=new_risk_perceptions,
            new_risk_aversions=new_risk_aversions,
            gender_movers=people['gender'],
            risk_aversion_movers=people['risk_aversion'],
            age_movers=people['age'],
            locations=self._locations,
            size=self._size,
            gender=self._gender,
            risk_aversion=self._risk_aversion,
            age = self._age,
            income_percentile = self._income_percentile,
            risk_perception=self._risk_perception,
            hh_risk_aversion=self._hh_risk_aversion,
            p_suitability=p_suitability,
            admin_indices=self.geom['properties']['gadm']['indices'],
            gt=self.geom['properties']['gadm']['gt']
        )
        self.initiate_household_attributes_movers(n_movers=new_income_percentiles.size) # Not the optimal placement. Maybe include this function in numba_add()
        
        assert (self.size != -1).any()
        self.population += people['from'].size

    @staticmethod
    @njit
    def move_numba(
        population,
        n: int,
        people_indices_per_household: np.ndarray,
        empty_index_stack: np.ndarray,
        empty_index_stack_counter: int,
        households_to_move: np.ndarray,
        n_movers: int,
        move_to_region: np.ndarray,
        admin_idx: int,
        locations: np.ndarray,
        size: np.ndarray,
        ead: np.ndarray,
        ead_dryproof: np.ndarray,
        gender: np.ndarray,
        risk_aversion: np.ndarray,
        age: np.ndarray,
        income_percentile: np.ndarray,
        income: np.ndarray,
        wealth: np.ndarray,
        risk_perception: np.ndarray,
        flood_timer: np.ndarray,
        adapt: np.ndarray,
        time_adapt: np.ndarray,
        decision_horizon: np.ndarray,
        hh_risk_aversion: np.ndarray,
        property_value: np.ndarray,
        amenity_value: np.ndarray


    ) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This function moves people out of this region.

        Args:
            n: Current number of households.
            people_indices_per_household: Maps the people in these households to positions in the per-person arrays.
            empty_index_stack: An array of indices with empty household ids, for example when a household moved.
            empty_index_stack_counter: The current stack index for `empty_index_stack`.
            households_to_move: the indices of the households to move.
            n_movers: The number of people moving.
            move_to_region: The region where those people will move to.
            admin_idx: The index of the current region.
            locations: The array that contains the current locations of the households.
            size: The array that contains the size of each of the households.
            gender: The gender of each of the people.
            risk_aversion: The risk aversion of the people.
            age: The age of the people.

        Returns:
            n: Current number of households (after the people moved).
            empty_index_stack_counter: The current stack index for `empty_index_stack`.
            from_region: Array of the region where the people moved from.
            to_region: Array of the region where people are moving to.
            household_id: Locally unique identifier that specifies the household id for each person.
            gender_movers: The gender of each of the movers.
            risk_aversion_movers: The risk aversion of each of the movers.
            age_movers: The age of each of the movers.
        """
        assert np.all(households_to_move[:-1] >= households_to_move[1:])  # ensure array is in descending order
        max_stack_counter = empty_index_stack.size
        from_region = np.full(n_movers, admin_idx, dtype=np.int32)
        to_region = np.full(n_movers, -1, dtype=np.int32)
        gender_movers = np.full(n_movers, -1, dtype=np.int8)      
        risk_aversion_movers = np.full(n_movers, -1, dtype=np.float32)
        age_movers = np.full(n_movers, -1, dtype=np.int8)        
        household_id = np.full(n_movers, -1, dtype=np.int32)
        
        # Household level attributes
        income_percentile_movers =  np.full(households_to_move.size, -1, dtype=np.int8)
        risk_perception_movers = np.full(households_to_move.size, -1, dtype=np.float32)
        household_income_movers = np.full(households_to_move.size, -1, dtype=np.float32)

        k = 0
        for i in range(households_to_move.size):
            household_to_move = households_to_move[i]
            households_size = size[household_to_move]
            move_to = move_to_region[i]

            # Household level attributes
            income_percentile_movers[i] = income_percentile[household_to_move]
            risk_perception_movers[i] = risk_perception[household_to_move]
            household_income_movers[i] =  income[household_to_move]

            for j in range(households_size):
                to_region[k] = move_to
                household_id[k] = i
                person_index = people_indices_per_household[household_to_move, j]  # index of the current mover
                assert person_index != -1
                gender_movers[k] = gender[person_index]  # set gender in move_dictionary
                risk_aversion_movers[k] = risk_aversion[person_index]  # set risk aversion in move_dictionary
                age_movers[k] = age[person_index]  # set risk aversion in move_dictionary

                # reset values for person
                gender[person_index] = -1
                risk_aversion[person_index] = -1
                age[person_index] = -1
                
                assert empty_index_stack[empty_index_stack_counter] != -1
                # increment self._empty_index_stack_counter
                empty_index_stack_counter += 1
                assert empty_index_stack_counter < max_stack_counter
                # just check if index stack is indeed empty
                assert empty_index_stack[empty_index_stack_counter] == -1
                empty_index_stack[empty_index_stack_counter] = person_index
                
                k += 1
            # Shifting household positions so the last positions in the array are removed
            size[household_to_move] = size[n-1]
            size[n-1] = -1
            locations[household_to_move] = locations[n-1]
            locations[n-1] = np.nan
            ead[household_to_move] = ead[n-1]
            ead[n-1] = -1
            ead_dryproof[household_to_move] = ead_dryproof[n-1]
            ead_dryproof[n-1] = -1
            income_percentile[household_to_move] = income_percentile[n-1]
            income_percentile[n-1] = -1
            income[household_to_move] = income[n-1]
            income[n-1] = -1
            wealth[household_to_move] = wealth[n-1]
            wealth[n-1] = -1
            risk_perception[household_to_move] = risk_perception[n-1]
            risk_perception[n-1] = -1
            flood_timer[household_to_move] = flood_timer[n-1]
            flood_timer[n-1] = -1
            adapt[household_to_move] = adapt[n-1]
            adapt[n-1] = 0
            time_adapt[household_to_move] = time_adapt[n-1]
            time_adapt[n-1] = -1
            decision_horizon[household_to_move] = decision_horizon[n-1]
            decision_horizon[n-1] = -1
            hh_risk_aversion[household_to_move] = hh_risk_aversion[n-1]
            hh_risk_aversion[n-1] = -1
            property_value[household_to_move] = property_value[n-1]
            property_value[n-1] = -1
            amenity_value[household_to_move] = amenity_value[n-1]
            amenity_value[n-1] = -1

            if household_to_move != n - 1:                
                # now we move the last household to the empty spot in the people indices
                people_indices_per_household[household_to_move] = people_indices_per_household[n-1]
            
            # and finally reset the last spot of the people_indices_per_household
            people_indices_per_household[n-1] = -1
            
            n -= 1
        population -= n_movers
        return population, n, empty_index_stack_counter, from_region, to_region, household_id, gender_movers, risk_aversion_movers, age_movers, income_percentile_movers, household_income_movers, risk_perception_movers
    
    def move(self):
        '''This function processes the individual household agent decisions. It calls the functions to calculate
        expected utility of stayin, adapting, and migrating to different regions. Agents that decide to move are
        then removed from the arrays'''
        # Reset counters
        self.n_moved_out_last_timestep = 0
        self.n_moved_in_last_timestep = 0
        self.people_moved_out_last_timestep = 0
        self.perc_people_moved_out = 0

        # Assign risk aversion sigma and time discounting preferences
        sigma = self.settings['decisions']['risk_aversion']
        r_time = self.settings['decisions']['time_discounting']
        
        # Run some checks to assert all households have attribute values
        # assert (sigma != -1).all()
        assert (self.income > -1).all()
        assert (self.wealth > -1).all()
        assert (self.risk_perception != -1).all()
        assert (self.decision_horizon != -1).all()
        assert (self.ead != -1).all()
        assert (self.hh_risk_aversion != -1).all() # Not used in decisions, all households currently have the same risk aversion setting (sigma).

        # Reset timer and adaptation status when lifespan of dry proofing is exceeded 
        self.adapt[self.time_adapt == self.settings['adaptation']['lifespan_dryproof']] = 0
        self.time_adapt[self.time_adapt == self.settings['adaptation']['lifespan_dryproof']] = -1 # People have to make adaptation choice again.
       
        # Only select region for calculations if agents experience flood risk
        if self.ead.size:
            # Transform all dictories to numpy array
            expected_damages = np.array(self.damages)
            expected_damages_adapt = np.array(self.damages_dryproof_1m)

            # Convert adaptation cost to annual cost based on loan duration and interest rate
            total_cost = self.settings['adaptation']['adaptation_cost']
            loan_duration = self.settings['adaptation']['loan_duration']
            r_loan =  self.settings['adaptation']['interest_rate']

            # Calculate annnual costs of adaptation loan based on interest rate and loan duration
            annual_cost = total_cost * (r_loan *( 1+r_loan) ** loan_duration/ ((1+r_loan)**loan_duration -1))
            
            # Fix risk perception at zero for a scenario of no dynamic behavior (not the best name)
            if not self.settings['general']['dynamic_behavior'] and not self.model.spin_up_flag:
                self.risk_perception *= 0

            # Collect all params in dictionary
            decision_params = {'loan_duration': self.settings['adaptation']['loan_duration'],
                'expendature_cap': self.settings['adaptation']['expenditure_cap'],
                'lifespan_dryproof' : self.settings['adaptation']['lifespan_dryproof'],
                'n_agents':  self.n,
                'sigma': sigma, 
                'wealth': self.wealth, 
                'income': self.income, 
                'amenity_value': self.amenity_value,
                'p_floods': 1/ self.rts, 
                'risk_perception': self.risk_perception, 
                'expected_damages': expected_damages,
                'expected_damages_adapt': expected_damages_adapt, 
                'adaptation_costs': np.full(self.n, annual_cost), 
                'adapted': self.adapt,
                'time_adapted' : self.time_adapt,
                'T': self.decision_horizon, 
                'r': r_time}
            
            # Determine EU of adaptation or doing nothing            
            EU_do_nothing = calcEU_no_nothing(**decision_params)
            

            # Determine EU of adaptation (set to -inf if we want to exclude this behavior)
            if self.settings['general']['include_adaptation'] or self.model.spin_up_flag:
                EU_adapt = calcEU_adapt(**decision_params)
            else:
                EU_adapt = calcEU_adapt(**decision_params)
                
                # Household can no longer implement adaptation after the spin-up period 
                EU_adapt[np.where(self.adapt != 1)] = -np.inf

            # Check output for missing data (if something went wrong in calculating EU)
            assert(EU_do_nothing != -1).any or (EU_adapt != -1).any()
            
            # Check if we want to model migration
            if (self.settings['general']['include_migration'] or self.model.spin_up_flag) and not self.model.calibrate_flag:
            
                # Select 25 closest regions
                regions_select = np.argsort(self.distance_vector)[1:self.settings['decisions']['regions_included_in_migr']+1] # closest regions and exclude own region
                
                
                # Determine EU of migration and which region yields the highest EU
                income_distribution_regions = np.array(self.agents.regions.income_distribution_region, dtype=np.int32)

                EU_migr_MAX, ID_migr_MAX = EU_migrate(
                    regions_select = regions_select,
                    n_agents = self.n,
                    sigma = sigma,
                    wealth = self.wealth,
                    income_distribution_regions = income_distribution_regions,
                    income_percentile = self.income_percentile,
                    amenity_value_regions = np.array(self.model.agents.regions.amenity_value_regions, dtype=np.int32),
                    distance = self.distance_vector,
                    T = self.decision_horizon,
                    r = r_time,
                    Cmax = self.settings['decisions']['migration']['max_cost'],
                    cost_shape =  self.settings['decisions']['migration']['cost_shape']
                )

                EU_migr_MAX = EU_migr_MAX
                
                # Create boolean array to indicate which households will adapt and migrate        
                EU_mig_bool = (EU_migr_MAX > EU_adapt) & (EU_migr_MAX > EU_do_nothing)

                # Intentions to behavior
                households_intenting_to_move =  np.where(EU_mig_bool == True)
                households_not_moving = np.random.choice(households_intenting_to_move[0], int((1-self.settings['decisions']['migration']['intention_to_behavior']) * households_intenting_to_move[0].size))
                EU_mig_bool[households_not_moving] = False

                EU_stay_adapt_bool = (EU_adapt > EU_do_nothing) & (EU_adapt >= EU_migr_MAX)
                self.adapt = EU_stay_adapt_bool * 1
                # Check which people will adapt and whether they made this decision for the first time
                pos_first_time_adapt = (self.adapt == 1) * (self.time_adapt == -1)

                # Set the timer for these people to 0
                self.time_adapt[pos_first_time_adapt] = 0

                # Update timer for next year
                self.time_adapt[self.time_adapt != -1] += 1
                        
                # Update the percentage of households implementing flood proofing
                # Check for missing data
                assert (self.adapt != -1).any()
                self.percentage_adapted = round(np.sum(self.adapt)/ len(self.adapt) * 100, 2)
                self.n_people_adapted = np.sum(self.adapt[self.adapt == 1])

                # Sum bool to get the number of households to move
                n_households_to_move = np.sum(EU_mig_bool)
                # print(f'households to move {n_households_to_move}')
            
            else:
                # Only compare EU of adapting vs. not adapting
                EU_stay_adapt_bool = (EU_adapt >= EU_do_nothing)
                self.adapt = EU_stay_adapt_bool * 1

                # Check which people will adapt and whether they made this decision for the first time
                pos_first_time_adapt = (self.adapt == 1) * (self.time_adapt == -1)

                # Set the timer for these people to 0
                self.time_adapt[pos_first_time_adapt] = 0

                # Update timer for next year
                self.time_adapt[self.time_adapt != -1] += 1

                # Update the percentage of households implementing flood proofing
                # Check for missing data
                assert (self.adapt != -1).any()
                self.percentage_adapted = round(np.sum(self.adapt)/ len(self.adapt) * 100, 2)
                self.n_people_adapted = np.sum(self.adapt[self.adapt == 1])

                n_households_to_move = 0
            


            if n_households_to_move > 0:          
                household_array = np.arange(self.n)
                households_to_move = household_array[EU_mig_bool]
                # print(self.geom['properties']['id'], n_households_to_move)
            else:
                # print('Region: ', self.geom['properties']['id'])  
                # print('No households to move')
                return None
        else:
            return None
        

        households_to_move.sort()
        households_to_move = households_to_move[::-1]

        households_sizes = self.size[households_to_move]
        n_movers = households_sizes.sum()
        
        self.n_moved_out_last_timestep = households_sizes.size
        self.people_moved_out_last_timestep = n_movers

        self.perc_people_moved_out = (n_movers/ self.population) * 100

        move_to_region = ID_migr_MAX[households_to_move]

        # Check if nobody is moving to their own region
        assert not any(move_to_region == self.admin_idx)

        self.population, self.n, self._empty_index_stack_counter, from_region, to_region, household_id, gender, risk_aversion, age, income_percentile, income, risk_perception = self.move_numba(
            self.population,
            self.n,
            self._people_indices_per_household,
            self._empty_index_stack,
            self._empty_index_stack_counter,
            households_to_move,
            n_movers,
            move_to_region,
            self.admin_idx,
            self._locations,
            self._size,
            self._ead,
            self._ead_dryproof,
            self._gender,
            self._risk_aversion,
            self._age,
            self._income_percentile,
            self._income,
            self._wealth,
            self._risk_perception,
            self._flood_timer,
            self._adapt,
            self._time_adapt,
            self._decision_horizon,
            self._hh_risk_aversion,
            self._property_value,
            self._amenity_value
        )

        # Create person income percentiles based on household IDs
        person_income_percentile = np.take(income_percentile, household_id)
        person_risk_perception = np.take(risk_perception, household_id)
        person_income = np.take(income, household_id)

        return {
            "from": from_region,
            "to": to_region,
            "household_id": household_id,
            "gender": gender,
            "risk_aversion": risk_aversion,
            "age": age,
            "income_percentile": person_income_percentile,
            "income": person_income,
            "risk_perception": person_risk_perception
        }

class InlandNode(HouseholdBaseClass, InlandNodeProperties):
    def __init__(self, model, agents, geom, distance_matrix, n_households_per_region, idx, max_household_size):
        self.model = model
        self.agents = agents
        self.geom = geom
        self.admin_idx = idx
        self.distance_vector = distance_matrix[idx]
        self.n_households_per_region = n_households_per_region
        assert not np.isnan(self.distance_vector).any()
        self.max_household_size = max_household_size

        data = self.model.data.population.sample_geom(self.geom)
        data = data.ravel()
        data = data[data != -1]
        population = np.round(np.sum(data)) 
        if np.isnan(population):
            population = 0
        self.population = population 

        self.n = self.population // ((max_household_size + 1)/ 2) # Average household size
        
        self.size = None # Required for export. Improve later
        self.income = None
        self.adapt = None
        self.risk_perception = None
        self.ead = None
        self.flood_timer = None
        self.income_percentile = np.array([-9999])

        HouseholdBaseClass.__init__(self, model, agents)


    def initiate_agents(self):
        return None

    def add(self, people):
        # Migration
        n_households = np.unique(people['household_id']).size
        self.n_moved_in_last_timestep = n_households
        self.n += n_households
        self.population += len(people['to'])
        
    def remove(self, n, n_movers):
        if n > self.n:
            raise ValueError("Cannot remove more households than present in the area") 
        if n_movers > self.population:
            raise ValueError("Cannot remove more people than present in the area")
        self.n_moved_out_last_timestep = n
        self.people_moved_out_last_timestep = n_movers
        self.n -= n
        self.population -= n_movers
    
        
    def move(self):
        '''This function moves people from the regional nodes. The gravity model function is called to calculate flow between regions.'''
        self.n_moved_out_last_timestep = 0
        self.n_moved_in_last_timestep = 0

        # Include migration or not
        if (self.settings['general']['include_migration'] or self.model.spin_up_flag) and not self.model.calibrate_flag:
            # Use gravity model to determine number of households migrating between regions
            gravity_dictionary = {}
            # Array to store generated household sizes
            household_sizes_all = np.array([], dtype=np.int16)
            n_people_to_move_dict = np.full(len(self.agents.regions.ids), -1, dtype=np.int16)
            n_people_to_move = 0

            # Itterate over each destination node       
      
            # Filter out floodplains
            # Make sure the array is in the correct order
            assert self.agents.regions.ids[-1].endswith('_flood_plain')

            # Filter
            destinations = [region for region in self.agents.regions.ids if not region.endswith('_flood_plain')]

            # Load social connectedness and filter admin
            for i, destination in enumerate(destinations):
                population_flood_plain_dest = 0
                dest_flood_plain = destination + '_flood_plain'

                household_sizes = np.array([])
                # Flows within the region are set to zero
                if i == self.admin_idx:
                    flow_people = 0
                    gravity_dictionary[i] = flow_people             
                    n_people_to_move_dict[i] = 0
                else:
                    flood_dest = False
                    if dest_flood_plain in self.agents.regions.ids:
                        flood_dest = True
                        dest_flood_plain_idx = self.agents.regions.ids.index(dest_flood_plain)
                        population_flood_plain_dest = self.agents.regions.population[dest_flood_plain_idx]

                    pop_i = self.population
                    pop_j = self.agents.regions.population[i] + population_flood_plain_dest
                    inc_i = self.income_region
                    inc_j = self.agents.regions.income_region[i]
                    
                    if any(self.model.data.coastal_admins[0] == self.admin_name):
                        coastal_i = 1
                    else:
                        coastal_i = 0
                    
                    if (any(destination == self.model.data.coastal_admins[0])):
                        coastal_j = 1
                    else:
                        coastal_j = 0

                    # convert distance to kilometers
                    distance = self.distance_vector[i]
                  
                    # apply gravity model to calculate people flow
                    flow_people = gravity_model(pop_i=pop_i, pop_j=pop_j,
                     inc_i=inc_i, inc_j=inc_j, coastal_i=coastal_i, coastal_j=coastal_j,
                     distance=distance)
                    

                    floodplain_flow = 0
                    inland_flow = flow_people

                    if flood_dest:
                        # Now determine the number of people moving to the department that has a floodplain and distribute based on populations
                        # frac of population living in floodplain
                        frac_pop = population_flood_plain_dest/ self.agents.regions.population[i]
                        
                        # flow towards the floodplain 
                        floodplain_flow = round(frac_pop * flow_people)

                        # flow towards inland portion 
                        inland_flow = flow_people - floodplain_flow
                        

                        n_people_to_move_dict[dest_flood_plain_idx] = floodplain_flow
                        
                        household_sizes = self.return_household_sizes(flow_people=floodplain_flow, max_household_size=self.max_household_size)
                        household_sizes_all = np.append(household_sizes_all, household_sizes) 

                        gravity_dictionary[dest_flood_plain_idx] = int(len(household_sizes[household_sizes != 0]))


                    # Add flow to total nr people moving out
                    n_people_to_move += flow_people
                    n_people_to_move_dict[i] = inland_flow

                    household_sizes = self.return_household_sizes(flow_people=inland_flow, max_household_size=self.max_household_size)
                    
                    # Store n_households moving out
                    gravity_dictionary[i] = int(len(household_sizes[household_sizes != 0]))
                    
                    # Append to array of all household sizes moving out 
                    household_sizes_all = np.append(household_sizes_all, household_sizes)         
            
                    
                        

            # Calculate total n households moving out
            n_households_to_move = len(household_sizes_all)

            # Initiate loop
            move_to_region_per_household = []

            # Loop to create household destinations
            for i in gravity_dictionary.keys():
                move_to_region_per_household = np.append(move_to_region_per_household, np.repeat(i, gravity_dictionary[i]))    
        
        
        else:
            n_households_to_move = None

        if not n_households_to_move:  # no household are moving
            return None

        # Create household attributes for move dictionary
        n_movers, to_region, household_id, gender, risk_aversion, age, income_percentile, income, risk_perception = self._generate_households(
            n_households_to_move=n_households_to_move,
            household_sizes=household_sizes_all,
            move_to_region_per_household = move_to_region_per_household,
            hh_risk_aversion = self.settings['decisions']['risk_aversion'],
            init_risk_perception = self.settings['flood_risk_calculations']['risk_perception']['min'],
            income_percentiles = nb.typed.List(self.model.agents.regions.income_percentiles_regions)
        )
        # Remove housheold and people from the population
        self.remove(n_households_to_move, n_people_to_move)

        # Calculate percentage moved out
        self.perc_people_moved_out = n_people_to_move / self.population * 100

        person_income_percentile = np.take(income_percentile, household_id)
        person_risk_perception =  np.take(risk_perception, household_id)
        person_income = np.take(income, household_id)
        # Create person level income percentile and risk perception for move dictionary

        # Return move dictionary
        return {
            "from": np.full(n_movers, self.admin_idx, dtype=np.int32),
            "to": to_region,
            "household_id": household_id,
            "gender": gender,
            "risk_aversion": risk_aversion,
            "age": age,
            "income_percentile": person_income_percentile,
            "income": person_income,
            "risk_perception": person_risk_perception,
        }
    
    def process_population_change(self):
        population_change, household_sizes = self.ambient_pop_change()
        self.population += population_change
        # Reshuffle households based on population change (improve later, although not relevant)
        self.n = np.int(self.population / (self.max_household_size+1/2))       


    def step(self):
        # pass
        if self.model.current_time.year > self.model.config['general']['start_time'].year:
            self.process_population_change()

class Nodes(AgentBaseClass, NodeProperties):
    # This class contains both the inland nodes (InlandNode) and coastal node (CoastalNode)
    def __init__(self, model, agents):
        self.model = model
        self.agents = agents
        self.initiate_agents()
        self._load_initial_state()

    def initiate_agents(self):
        self.geoms = self.model.area.geoms['admin']
        self.n = len(self.geoms)
        self._initiate_attributes()

        max_household_size = 6
        
        self.household = {}
        self.aggregate_household = {}
        self.all_households = []
        j = 0
        n_households_per_region = np.full(len(self.geoms), 0, dtype=np.int32)
        for i, geom in enumerate(self.geoms):
            
            # All nodes
            ID = geom['properties']['id']

            # Coastal nodes 
            if ID.endswith('flood_plain') & self.model.config['general']['create_agents']== True:
                init_folder = os.path.join("DataDrive", "SLR", f"households_gadm_{self.model.args.admin_level}_{self.model.config['general']['start_time'].year}", self.model.config['general']['size'], ID.replace('_flood_plain', ''))
                locations_fn = os.path.join(init_folder, "locations.npy")
                if os.path.exists(locations_fn):
                    redundancy = np.load(locations_fn).size * 100  # redundancy of 1000%
                else:
                    redundancy = 0  # nobody should be moving to areas with no population
                household_class = CoastalNode(
                    self.model,
                    self.agents,
                    idx=i,
                    geom=geom,
                    distance_matrix=self.distance_matrix,
                    n_households_per_region=n_households_per_region,
                    init_folder=init_folder,
                    max_household_size=max_household_size,
                    redundancy=redundancy,
                    person_reduncancy=int(redundancy * (max_household_size // 2))
                )
                self.household[ID] = household_class
            else:

                household_class = InlandNode(
                    self.model,
                    self.agents,
                    idx=i,
                    geom=geom,
                    distance_matrix=self.distance_matrix,
                    n_households_per_region=n_households_per_region,
                    max_household_size=max_household_size
                )
                j += 1
                self.aggregate_household[ID] = household_class
            self.all_households.append(household_class)
        
        self.model.logger.info(f'Created {sum(household.n for household in self.household.values())} households')
        self.model.logger.info(f'Created {sum(household.n for household in self.aggregate_household.values())} aggregrate households')
        

    def _load_initial_state(self):
        return None

    def _get_distance_matrix(self):
        centroids = self.centroids
        
        # Convert (back) to geopandas points
        gpd_points = gpd.points_from_xy(centroids[:,0], centroids[:,1], crs='EPSG:4326')

        # Project points in world Mollweide (?)
        gpd_points = gpd_points.to_crs('ESRI:54009')

        # Extract x and y   
        x = gpd_points.x
        y = gpd_points.y

        # Stack into dataframe and export distance matrix
        projected_centroids = np.column_stack((x, y))
        return sdistance_matrix(projected_centroids, projected_centroids) /1000 # Devide by 1000 to derive km
    
    def _initiate_attributes(self):
        self.distance_matrix = self._get_distance_matrix()

    def merge_move_dictionary(self, move_dictionaries):
        merged_move_dictionary = {}
        for key in move_dictionaries[0].keys():
            # Household_ids are determined in the origin region. The procedure below ensures that
            # all household_ids are unique across the merged dictionary.
            if key == 'household_id':
                c = 0
                household_ids_per_region = [d[key] for d in move_dictionaries]
                household_ids_per_region_corrected = []
                for household_ids in household_ids_per_region:
                    if household_ids.size > 0:
                        household_ids_per_region_corrected.append(household_ids + c)
                        c += household_ids[-1] + 1
                merged_move_dictionary[key] = np.hstack(household_ids_per_region_corrected)
                self.model.logger.info(f"Moving {len(merged_move_dictionary[key])} agents")
            else:
                merged_move_dictionary[key] = np.hstack([d[key] for d in move_dictionaries])
        return merged_move_dictionary

    def step(self):
        for households in self.all_households:
            households.step()
            
        move_dictionaries = []
        for region in self.all_households:
            move_dictionary = region.move()

            if move_dictionary:
                move_dictionaries.append(move_dictionary)
        if move_dictionaries:
            # merge movement data of different regions in 1 dictionary
            merged_move_dictionary = self.merge_move_dictionary(move_dictionaries)
            if households.settings['general']['export_move_dictionary']:
                merged_pd = pd.DataFrame(merged_move_dictionary)#.to_csv(os.path.join(self.model.config['general']['report_folder'], f'move_dictionary_{self.model.current_time.year}.csv'))
                
                unique_household_ids = merged_pd.groupby('household_id').mean().drop(columns = ['gender', 'age'])     

                fn = os.path.join(self.model.config['general']['report_folder'], 'move_dictionaries')
                if not os.path.exists(fn):
                    os.makedirs(fn)
                
                from_region = [self.model.agents.regions.ids[int(i)] for i in unique_household_ids['from']]
                to_region = [self.model.agents.regions.ids[int(i)] for i in unique_household_ids['to']]
                unique_household_ids['from'] = from_region
                unique_household_ids['to'] = to_region
                coastal = [coast_key for coast_key in list(unique_household_ids['from']) if coast_key.endswith('flood_plain')]
                unique_household_ids_filt = unique_household_ids[unique_household_ids['from'].isin(coastal)]
                unique_household_ids_filt.to_csv(os.path.join(fn, f'move_dictionary_housholds_{self.model.current_time.year}.csv'))

            # split the movement data by destination region and send data to region

            sort_idx = np.argsort(merged_move_dictionary['to'], kind='stable')
            for key, value in merged_move_dictionary.items():
                merged_move_dictionary[key] = value[sort_idx]
            move_to_regions, start_indices = np.unique(merged_move_dictionary['to'], return_index=True)
            end_indices = np.append(start_indices[1:], merged_move_dictionary['to'].size)
       
            for region_idx, start_idx, end_idx in zip(move_to_regions, start_indices, end_indices):
                move_data_per_region = {
                    key: value[start_idx: end_idx]
                    for key, value in merged_move_dictionary.items()
                }

                self.all_households[region_idx].add(move_data_per_region)   
       
        # Export all agents to csv
        if households.settings['general']['export_agents']:
            export_agents(agent_locs = self.agents.regions.agent_locations, agent_size = self.agents.regions.household_sizes,
                agent_adapt = self.agents.regions.household_adapted, agent_income=self.agents.regions.household_incomes,
                agent_risk_perception = self.agents.regions.household_risk_perception, agent_ead=self.agents.regions.household_ead,
                agent_since_flood = self.agents.regions.since_flood,
                year = self.model.current_time.year, 
                export = True,
                report_folder=self.model.config['general']['report_folder'])   


        if households.settings['general']['export_matrix'] and move_dictionaries:
            # Export matrix to csv
            export_folder = os.path.join(self.model.config['general']['report_folder'], 'migration_matrices')
            export_matrix(geoms = self.model.agents.regions.geoms, 
                        dest_folder=export_folder, 
                        move_dictionary=merged_move_dictionary,
                        year=self.model.current_time.year)

class Agents:
    def __init__(self, model):
        self.model = model
        self.agent_types = []
        self.regions = Nodes(model, self)
    def step(self):     
        self.regions.step()
        if not self.model.spin_up_flag or not self.model.args.headless:
            self.population_data = WorldPopProspectsChange(initial_figures=self.model.data.nat_pop_change,
             HistWorldPopChange = self.model.data.HistWorldPopChange, WorldPopChange = self.model.data.WorldPopChange, 
             population=self.regions.population, admin_keys=self.regions.ids, year = self.model.current_time.year)