import numpy as np
from numba import njit
import numba as nb
import os
from agents.base_nodes import HouseholdBaseClass
from honeybees.library.raster import  pixel_to_coord
from honeybees.library.neighbors import find_neighbors
from scipy import interpolate
from hazards.flooding.flood_risk import FloodRisk
from decision_module import calcEU_no_nothing, calcEU_adapt, EU_migrate
from hazards.erosion.shoreline_change import find_indices_closest_segment
from agents.node_properties import CoastalNodeProperties
from agents.coastal_amenities import total_amenity



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
        HouseholdBaseClass.__init__(self, model, agents)
        np.random.seed(0)
        self.model.set_seed(0)


    def initiate_agents(self):
        self._initiate_locations()
        self._initiate_household_attributes()
        self._initiate_person_attributes()
        self._initiate_shoreline_change_admin()
        self._initiate_utility_surface()
        self.calculate_flood_risk()
        # self.process_coastal_erosion()

    def _initiate_utility_surface(self):
        # assign suitability matrix
        admin_indices = self.geom['properties']['gadm']['indices']

        # Get coords associated with pixels in admin region
        px = admin_indices[1][:] + 0.5
        py = admin_indices[0][:] + 0.5

        locations_admin_cells = pixel_to_coord(px=px, py=py, gt=self.geom['properties']['gadm']['gt'])
        self.locations_admin_cells = np.stack([locations_admin_cells[0], locations_admin_cells[1]], axis = 1)

        # Sample values for all admin cells
        ead_array_admin_cells = self.model.data.ead_map.sample_coords(self.locations_admin_cells, dt = self.model.current_time) # only done once. Fix creation of flood risk class
        dist_to_coast_admin_cells = self.model.data.distance_to_coast.sample_coords(self.locations_admin_cells)
        
        # Sample traveltime to city density (static)
        

        # # Adjust for min and max distance and apply amenity function
        dist = np.array(self.model.data.coastal_amenity_functions['dist2coast'].index)
        amenity_factor = np.array(self.model.data.coastal_amenity_functions['dist2coast']['premium'])
        calc_amenity = interpolate.interp1d(x = dist, y = amenity_factor)
        dist_to_coast_admin_cells = np.maximum(min(dist),  dist_to_coast_admin_cells)
        dist_to_coast_admin_cells = np.minimum(max(dist),  dist_to_coast_admin_cells)

        # # Calculate utility of all cells
        self.coastal_amenity_cells = calc_amenity(dist_to_coast_admin_cells) * self.wealth.mean()
        self.damages_coastal_cells = ead_array_admin_cells * self.property_value.mean()
        
        # Extract urban classes for urban mask 
        urban_classes = self.model.data.SMOD.sample_coords(self.locations_admin_cells) 

        # Urban mask
        smod_class = 13 #“Rural cluster grid cell”, if the cell belongs to a Rural Cluster spatial entity;
        self.smod_mask = np.where(urban_classes >= smod_class) # all urban intenitities including and above the rural urban claster class

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

    def _initiate_shoreline_change_admin(self):
        names = [self.geom_id, self.geom_id[:-12]]
        self.segment_indices_within_admin = np.array(self.model.data.beach_ids.loc[self.model.data.beach_ids['keys'].isin(names)].index)

            
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

        self.total_shoreline_change_admin = 0

        # Initiate wealth, income, flood experience and adaptation status
        self._ead = np.full(self.max_n, -1)
        self._ead_dryproof = np.full(self.max_n, -1,  dtype=np.float32)
        self._adapt = np.full(self.max_n, -1,  dtype=np.float32)
        self._time_adapt = np.full(self.max_n, -1)
        self._income = np.full(self.max_n, -1)
        self._flooded = np.full(self.max_n, -1)
        self._flood_count = np.full(self.max_n, -1)
        self._amenity_value = np.full(self.max_n, -1)


        self.flooded = 0
        self.flood_count = 0

        # Position in the income distribution (related to eduction/ age etc)
        self._income_percentile = np.full(self.max_n, -1)
        self.income_percentile = np.random.randint(1, 100, self.n)

        self._income = np.full(self.max_n, -1)
        self.income = np.percentile(self.income_distribution_region, self.income_percentile)
                          
        self._property_value = np.full(self.max_n, -1)
        self.property_value = self.model.settings['flood_risk_calculations']['property_value']

        # Create dict of income/ wealth ratio and interpolate
        perc = np.array([0, 20, 40, 60, 80, 100])
        ratio = np.array([0, 1.06, 4.14, 4.19, 5.24, 6])
        self.income_wealth_ratio = interpolate.interp1d(perc, ratio)

        self._wealth =  np.full(self.max_n, -1)
        self.wealth =  self.income_wealth_ratio(self.income_percentile) * self.income
        self.wealth[self.wealth < self.property_value] = self.property_value[self.wealth < self.property_value] 
        
        self._decision_horizon = np.full(self.max_n, -1)
        self.decision_horizon = self.model.settings['decisions']['decision_horizon']
        
        self._hh_risk_aversion = np.full(self.max_n, -1, dtype= np.float32)
        self.hh_risk_aversion = self.model.settings['decisions']['risk_aversion']

        self._risk_perception = np.full(self.max_n, -1, dtype = np.float32)
        self.risk_perception = self.model.settings['flood_risk_calculations']['risk_perception']['min']

        self._flood_timer = np.full(self.max_n, -1, dtype = np.int32)
        self.flood_timer = 99 # Assure new households have min risk perceptions

        self._shoreline_change_agent = np.full(self.max_n, -1, dtype = np.float32)
        self.shoreline_change_agent = 0

        self._beach_proximity_bool =  np.full(self.max_n, False, dtype = bool)
                
        self.average_amenity_value = 0

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
        
        hh_risk_aversion = np.full(self.n, self.model.settings['decisions']['risk_aversion'], dtype=np.float32)
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

    def update_utility_surface(self):
        # Read ead from flood risk class instance
        ead_array_admin_cells = self.model.flood_risk.ead_admin_cells[self.geom_id]      
        self.damages_coastal_cells = ead_array_admin_cells * self.property_value.mean()

    def calculate_flood_risk(self): # Change this to calculate risk
      
        self.water_level = FloodRisk.sample_water_level(
            locations = self.locations,
            return_periods = np.array([key for key in  self.model.data.inundation_maps_hist.keys()]),
            fps = self.model.settings['flood_risk_calculations']['flood_protection_standard'],
            inundation_maps = [self.model.data.inundation_maps_hist, self.model.data.inundation_maps_2080],
            rcp = self.model.args.rcp,
            start_year = self.model.config['general']['start_time'].year,
            current_year = self.model.current_time.year)
       
        self.damages, self.damages_dryproof_1m, self.ead, self.ead_dryproof = FloodRisk.calculate_ead(
            n_agents = self.n,
            water_level = self.water_level,
            damage_curves = self.model.data.curves,
            property_value = self.property_value,
            return_periods = np.array([key for key in  self.model.data.inundation_maps_hist.keys()]))

        self.flooded, self.flood_count, self.risk_perception, self.flood_timer, self.flood_tracker = FloodRisk.stochastic_flood(
            water_levels = self.water_level,
            return_periods = np.array([key for key in  self.model.data.inundation_maps_hist.keys()]),
            flooded = self.flooded,
            flood_count = self.flood_count,
            risk_perceptions = self.risk_perception,
            flood_timer = self.flood_timer,
            risk_perc_min = self.model.settings['flood_risk_calculations']['risk_perception']['min'],
            risk_perc_max = self.model.settings['flood_risk_calculations']['risk_perception']['max'],
            risk_decr = self.model.settings['flood_risk_calculations']['risk_perception']['coef'],
            settings = self.model.settings['general']['flood'],
            current_year=self.model.current_time.year,
            spin_up_flag = self.model.spin_up_flag,
            flood_tracker= self.flood_tracker)       

        self.ead_total = np.sum(self.ead[self.adapt == 0]) + np.sum(self.ead_dryproof[self.adapt == 1])

    def process_coastal_erosion(self):               
        self.beach_proximity_bool = self.model.data.coastal_raster.sample_coords(self.locations) == 1
        self.people_near_beach = np.sum(self.size[self.beach_proximity_bool])
        self.households_near_beach = np.sum(self.beach_proximity_bool)
        segment_location_admin = self.model.agents.beaches.segment_locations[self.segment_indices_within_admin]
        beach_width_agent = np.full(self.n, 0, dtype=np.float32)

        # Find beach segment closts to agent
        if segment_location_admin.size > 0:
            segment_indices_agent = find_indices_closest_segment(self.locations, segment_location_admin, self.beach_proximity_bool)
            beach_width_agent[self.beach_proximity_bool] = self.model.agents.beaches.beach_width[segment_indices_agent]
        else: 
            segment_indices_agent = np.array([])
        #extract beach width experienced by agent

        # Calculate amenities based on beach proximity and distance to coast
        self.amenity_value, beach_amenity = total_amenity(
            coastal_amenity_functions = self.model.data.coastal_amenity_functions, 
            beach_proximity_bool = self.beach_proximity_bool, 
            dist_to_coast_raster = self.model.data.distance_to_coast,
            agent_locations = self.locations, 
            beach_width = beach_width_agent,
            agent_wealth = self.wealth,
        )
        
        self.add_segment_value(segment_indices_agent, beach_amenity)
        # self.average_amenity_value = np.min([np.median(self.amenity_value), 15_000])

    def add_segment_value(self, segment_indices_agent, beach_amenity):
        # Sum beach amenity value for each coastal segment (vectorize this, although quite fast already)
        if segment_indices_agent.size > 0:
            for agent, segment in enumerate(segment_indices_agent):
                self.model.agents.beaches.value_segment[segment] += beach_amenity[agent]
        
        
    def initiate_household_attributes_movers(self, n_movers):
        '''This function assigns new household attributes to the households that moved into a floodplain. It takes all arrays and fills in the missing data based on sampling.'''
        
        assert n_movers == 0 or self.income[-n_movers] == -1 
        
        # Sample income percentile for households moving in from inland node or natural pop change
        # Find neighbors for newly generated households
        new_households = np.where(self.income_percentile == -99)[0]   

        # Fill
        neighbor_households_1km = find_neighbors(locations=self.locations, radius=1_000, n_neighbor=30, bits=32, search_ids=new_households)
        for i, household in enumerate(new_households):
            
            # Get neighbors
            neighboring_hh = neighbor_households_1km[i, :]
            
            if neighboring_hh.size == 0:
                neighbor_households_3km = find_neighbors(locations=self.locations, radius=3_000, n_neighbor=30, bits=32, search_ids=new_households)           

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
        self.property_value[-n_movers:self.n] = self.model.settings['flood_risk_calculations']['property_value']

        # Set wealth to never be lower than property value
        self.wealth[self.wealth < self.property_value] = self.property_value[self.wealth < self.property_value] 
        
        # Set decision horizon and flood timer
        self.decision_horizon[self.decision_horizon == -1] = self.model.settings['decisions']['decision_horizon']
        self.flood_timer[self.flood_timer == -1] = 99
       
        # self.amenity_value = 10_000
        # self.average_amenity_value = 0

        # Reset flood status to 0 for all households (old and new) and adaptation to 0 for new households
        self.adapt[-n_movers:] = 0
        self.flood_count[-n_movers:] = 0
        
    def process_population_change(self):
        population_change, household_sizes = self.ambient_pop_change()
        
        households_to_remove = []
        if population_change < 0 and self.population > abs(population_change):

            # Select households to remove from
            # Sample the households that are due for removal from self.size
            individuals_removed = 0
            households_to_remove = []
            i = 0

            # iterate while number of individuals removed does not meet projections or iterations exceed limit
            while individuals_removed < abs(population_change) and i < 1E6:
                # print(f'iteration {i}')
                household = np.random.randint(0, self.size.size)
                if household not in households_to_remove:
                    individuals_removed += self.size[household]
                    households_to_remove.append(household)
                i += 1
            households_to_remove = np.sort(households_to_remove)[::-1]          

            n_movers = np.sum(self.size[households_to_remove])
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
                hh_risk_aversion = self.model.settings['decisions']['risk_aversion'],
                init_risk_perception = self.model.settings['flood_risk_calculations']['risk_perception']['min'],
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
        if self.model.current_time.year > self.model.config['general']['start_time'].year and self.model.settings['general']['include_ambient_pop_change']:
            self.process_population_change()
        self.load_timestep_data()
        self.process()
        if self.model.config['general']['create_agents']:
            self.calculate_flood_risk()
            self.process_coastal_erosion()
            self.update_utility_surface()

    @staticmethod
    @njit
    def add_numba(
        n: int,
        cells_to_assess,
        smod_mask,
        damages_coastal_cells,
        coastal_amenity_cells,
        amenity_weight,
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

        for index_first_person_in_household, new_household_size, new_income_percentile, new_risk_perception, new_risk_aversion in zip(index_first_persons_in_household, new_household_sizes, new_income_percentiles, new_risk_perceptions, new_risk_aversions):
            n += 1
            assert n > 0
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
            # Apply a random sample of 10 cells:
            sample_size = np.minimum(10, smod_mask[0].size)
            cells_to_assess = np.random.choice(smod_mask[0], size = sample_size, replace=False)
            
            # select cells to assess by the agent
            damages_coastal_cells_agent = damages_coastal_cells[cells_to_assess]
            coastal_amenity_cells_agent = coastal_amenity_cells[cells_to_assess]

            # multiply expected damages with risk perception of the households
            agent_risk_perception = np.maximum(0.01, new_risk_perception) # account for risk perceptions of -1, will be set later (household not moving in from other coastal nodes)
            damages_coastal_cells_agent = agent_risk_perception * damages_coastal_cells_agent 

            # multiply amenity value with amenity weight
            coastal_amenity_cells_agent = coastal_amenity_cells_agent * amenity_weight

            # calculate utility cells
            utility_cells = coastal_amenity_cells_agent - damages_coastal_cells_agent

            # pick cell with highest utility
            cell = np.argsort(utility_cells)[-1:]

            if cell.size == 0: # if no urban cells in floodplain distribute randomly
                cell = np.random.randint(0, admin_indices[0].size)
            else:
                cell = cell[0] # array should be unpacked for njit

            # Allocate randomly in 1km2 grid cell (otherwise all agents the center of the cell) # Are actually positioned at the top left corner
            px = admin_indices[1][cell] + np.random.random()
            py = admin_indices[0][cell] + np.random.random()

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

        self.n, self._empty_index_stack_counter = self.add_numba(
            n=self.n,
            cells_to_assess = self.model.settings['decisions']['migration']['cells_to_assess'],
            smod_mask=self.smod_mask,
            damages_coastal_cells=self.damages_coastal_cells,
            coastal_amenity_cells=self.coastal_amenity_cells,
            amenity_weight=self.model.settings['decisions']['migration']['amenity_weight'],
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
        # Average household income and update GINI, store for export
        self.average_household_income = self.income.mean()
        
        # Reset counters
        self.n_moved_out_last_timestep = 0
        self.n_moved_in_last_timestep = 0
        self.people_moved_out_last_timestep = 0
        self.perc_people_moved_out = 0

        # Assign risk aversion sigma and time discounting preferences
        sigma = self.model.settings['decisions']['risk_aversion']
        r_time = self.model.settings['decisions']['time_discounting']
        
        # Run some checks to assert all households have attribute values
        # assert (sigma != -1).all()
        assert (self.income > -1).all()
        assert (self.wealth > -1).all()
        assert (self.risk_perception != -1).all()
        assert (self.decision_horizon != -1).all()
        assert (self.ead != -1).all()
        assert (self.hh_risk_aversion != -1).all() # Not used in decisions, all households currently have the same risk aversion setting (sigma).

        # Reset timer and adaptation status when lifespan of dry proofing is exceeded 
        self.adapt[self.time_adapt == self.model.settings['adaptation']['lifespan_dryproof']] = 0
        self.time_adapt[self.time_adapt == self.model.settings['adaptation']['lifespan_dryproof']] = -1 # People have to make adaptation choice again.
       
        # Only select region for calculations if agents experience flood risk
        if self.ead.size:
            # Transform all dictories to numpy array
            expected_damages = np.array(self.damages)
            expected_damages_adapt = np.array(self.damages_dryproof_1m)

            # Convert adaptation cost to annual cost based on loan duration and interest rate
            total_cost = self.model.settings['adaptation']['adaptation_cost']
            loan_duration = self.model.settings['adaptation']['loan_duration']
            r_loan =  self.model.settings['adaptation']['interest_rate']

            # Calculate annnual costs of adaptation loan based on interest rate and loan duration
            annual_cost = total_cost * (r_loan *( 1+r_loan) ** loan_duration/ ((1+r_loan)**loan_duration -1))
            
            # Fix risk perception at zero for a scenario of no dynamic behavior (not the best name)
            if not self.model.settings['general']['dynamic_behavior'] and not self.model.spin_up_flag:
                self.risk_perception *= 0

            # Collect all params in dictionary
            decision_params = {'loan_duration': self.model.settings['adaptation']['loan_duration'],
                'expendature_cap': self.model.settings['adaptation']['expenditure_cap'],
                'lifespan_dryproof' : self.model.settings['adaptation']['lifespan_dryproof'],
                'n_agents':  self.n,
                'sigma': sigma, 
                'wealth': self.wealth, 
                'income': self.income, 
                'amenity_value': self.amenity_value,
                'amenity_weight': self.model.settings['decisions']['migration']['amenity_weight'],
                'p_floods': 1/ np.array([key for key in self.model.data.inundation_maps_hist.keys()]), 
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
            if self.model.settings['general']['include_adaptation'] or self.model.spin_up_flag:
                EU_adapt = calcEU_adapt(**decision_params)
            else:
                EU_adapt = calcEU_adapt(**decision_params)
                
                # Household can no longer implement adaptation after the spin-up period 
                EU_adapt[np.where(self.adapt != 1)] = -np.inf

            # Check output for missing data (if something went wrong in calculating EU)
            assert(EU_do_nothing != -1).any or (EU_adapt != -1).any()
            
            # Check if we want to model migration
            if (self.model.settings['general']['include_migration'] or self.model.spin_up_flag) and not self.model.calibrate_flag:
            
                # Select 25 closest regions
                regions_select = np.argsort(self.distance_vector)[1:self.model.settings['decisions']['regions_included_in_migr']+1] # closest regions and exclude own region
                
                
                # Determine EU of migration and which region yields the highest EU
                income_distribution_regions = np.array(self.agents.regions.income_distribution_region, dtype=np.int32)

                EU_migr_MAX, ID_migr_MAX = EU_migrate(
                    regions_select = regions_select,
                    n_agents = self.n,
                    sigma = sigma,
                    wealth = self.wealth,
                    income_distribution_regions = income_distribution_regions,
                    income_percentile = self.income_percentile,
                    amenity_value_regions = self.model.agents.regions.amenity_value_regions,
                    amenity_weight=self.model.settings['decisions']['migration']['amenity_weight'],
                    distance = self.distance_vector,
                    T = self.decision_horizon,
                    r = r_time,
                    Cmax = self.model.settings['decisions']['migration']['max_cost'],
                    cost_shape =  self.model.settings['decisions']['migration']['cost_shape']
                )

                EU_migr_MAX = EU_migr_MAX
                
                # Create boolean array to indicate which households will adapt and migrate        
                EU_mig_bool = (EU_migr_MAX > EU_adapt) & (EU_migr_MAX > EU_do_nothing)

                # Intentions to behavior
                households_intenting_to_move =  np.where(EU_mig_bool == True)
                households_not_moving = np.random.choice(households_intenting_to_move[0], int((1-self.model.settings['decisions']['migration']['intention_to_behavior']) * households_intenting_to_move[0].size), replace = False)
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
                self.n_households_adapted = np.sum(self.adapt[self.adapt == 1])

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
                self.n_households_adapted = np.sum(self.adapt[self.adapt == 1])

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
