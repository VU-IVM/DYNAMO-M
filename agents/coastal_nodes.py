import numpy as np
from numba import njit
import os
from agents.base_nodes import HouseholdBaseClass
from honeybees.library.raster import pixel_to_coord, coords_to_pixels
from scipy import interpolate
from hazards.flooding.flood_risk import FloodRisk
from agents.node_properties import CoastalNodeProperties

class CoastalNode(HouseholdBaseClass, CoastalNodeProperties):
    attributes_to_cache = ()

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

    def __init__(
            self,
            model,
            agents,
            geom: dict,
            distance_matrix: np.ndarray,
            n_households_per_region: np.ndarray,
            idx: int,
            redundancy: int,
            person_reduncancy: int,
            init_folder: str,
            max_household_size: int):
        
        self.n_households_exposed = 0
        self.redundancy = redundancy
        self.person_reduncancy = person_reduncancy
        self.geom = geom
        self.admin_idx = idx
        self.distance_vector = distance_matrix[idx]
        self.n_households_per_region = n_households_per_region
        self.init_folder = init_folder
        self.max_household_size = max_household_size
        HouseholdBaseClass.__init__(self, model, agents)
        
    def move_memmaps(self):
        pass

    def update_scratch_folder(self):
        self.cache_file = os.path.join(self.model.args.low_memory_mode_folder, 'spin_up', self.model.args.rcp, f"{self.geom_id}.npz")  

    def update_cache_path(self):
        self.cache_file = os.path.join(self.model.args.low_memory_mode_folder, f'{self.model.args.rcp}_{self.model.args.ssp}', f"{self.geom_id}.npz")
        folder_name = os.path.dirname(self.cache_file)
        os.makedirs(folder_name, exist_ok=True)

    def initiate_memmap_arrays(self):
        pass

    def initiate_cache_file(self):
        if self.model.args.run_from_cache and self.model.spin_up_flag:
            self.cache_file = os.path.join(self.model.args.low_memory_mode_folder, 'spin_up', self.model.args.rcp, f"{self.geom_id}.npz")  
            folder_name = os.path.dirname(self.cache_file)
            os.makedirs(folder_name, exist_ok=True)
       
        elif self.model.args.low_memory_mode:        
            self.cache_file = os.path.join(self.model.args.low_memory_mode_folder, self.model.args.rcp, f"{self.geom_id}.npz")
            folder_name = os.path.dirname(self.cache_file)
            os.makedirs(folder_name, exist_ok=True)
        else:
            self.cache_file = None

    def save_arrays_to_npz(self):
        arrays_to_save = {}
        attributes_to_delete = []
        memmap_meta_data = []

        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, np.ndarray) and attr_name in self.attributes_to_cache:
                arrays_to_save[attr_name] = attr_value
                attributes_to_delete.append(attr_name)
            elif isinstance(attr_value, np.memmap):
                attr_value.flush()
                # if attr_name != 'water_levels_admin_cells':
                attributes_to_delete.append(attr_name)
                memmap_meta_data.append({'filename': attr_value.filename, 'shape': attr_value.shape, 'dtype': attr_value.dtype, 'attr_name': attr_name})

        for attr_name in attributes_to_delete:
            delattr(self, attr_name)
        if len(arrays_to_save) > 0:
            np.savez(self.cache_file, **arrays_to_save)
        
        self.memmap_meta_data = memmap_meta_data

    def load_arrays_from_npz(self):
        if len(self.attributes_to_cache) > 0:
            loaded_arrays = np.load(self.cache_file)
            for attr_name in loaded_arrays.files:
                setattr(self, attr_name, loaded_arrays[attr_name])
        
        for memmap in self.memmap_meta_data:
            if not os.path.exists(memmap['filename']):
                # scratch is different, update filepath
                attr_name = memmap['attr_name']
                if attr_name.startswith('_'):
                    attr_name = attr_name[1:]
                memmap['filename'] = os.path.join(self.model.args.low_memory_mode_folder, 'spin_up', 'rcp4p5', f'{self.geom_id}_{attr_name}.dat')
            array = np.memmap(memmap['filename'], dtype=memmap['dtype'], mode='r+', shape=memmap['shape'])
            setattr(self, memmap['attr_name'], array)
        # load memmaps here

    @staticmethod
    @njit
    def get_agent_indice_admin_cells(agent_indices, region_indices):
        '''This function is used to get the indice of the agent cell relative to all admin cells based on agent location'''
        # simple function to get the agent cell indice relative to all admin cells based on agent location
        indice_cell_agent = np.full(agent_indices[0].size, -1, np.int32)
        
        i = 0
        # get loc in admin indices
        for x, y in zip(agent_indices[0], agent_indices[1]):
            indice_cell =  np.where(np.logical_and(region_indices[0] == y, region_indices[1] ==x))[0]
            if indice_cell.size > 0:
                indice_cell_agent[i] = indice_cell[0]
            else:
                # if not in gadm indices find closest cell
                indice_cell_agent[i]  = np.argmin((region_indices[0] - y) **2 + (region_indices[1] - x) **2)
            i += 1
        return indice_cell_agent
          
    def initiate_agents(self):
        '''This function initiates the agents (and admin geom cells) in the region.'''
        self.initiate_cache_file()           
        self.initate_admin_geom()
        self._initiate_locations()
        self._initiate_household_attributes()
        self._initiate_person_attributes()
        self.calculate_flood_risk()
        if self.model.args.low_memory_mode:
            self.save_arrays_to_npz()
        self._initiate_max_household_density_admin()
        self.create_mask_household_allocation()
        self._initiate_info_floodplain()
        self.calculate_coastal_amenity_values()

    def initate_admin_geom(self):
        '''This function initiates several (index) arrays pertaining to the cells within the admin geom.'''
        self._load_damage_data_node()
        self._initiate_coastal_fps()
        self._initiate_admin_cell_coords()
        self._initiate_admin_cell_dist_to_coast()
        self._initiate_beach_proximity_bool_admin()
        self._initiate_dike_lenghts()
        self._initiate_water_level_polynomials_admin()
        self._initiate_dike_lenghts()
        self._initiate_admin_coastal_dikes()
        self._initiate_reporter_values()
   
    def _initiate_reporter_values(self):
        self.annual_summed_adaptation_costs = 0
    
        # self._initiate_utility_surface() # I want to have this here. Still working on this.
    def _initiate_info_floodplain(self):
        if self.beach_proximity_bool.sum() > 0:
            self.frac_coastal_cells = self.beach_proximity_bool.sum()/self.n
        else: self.frac_coastal_cells = 0

    def _initiate_coastal_fps(self):
        # read_fps_FLOPROS(self)
        if self.model.settings['flood_risk_calculations']['flood_protection_standard'] == 'FLOPROS':
            # sample from flopros and assign
            data = self.model.data.flopros.sample_geom(self.geom)
            data = data.ravel()
            data = data[data != -1]

            # average_fps in cells
            if data.size > 0 and np.min(data)> 0:
                # assume that the maximum fps in a coastal geom is valid for the entire geom
                average_fps = int(np.median(data))
                # adjust for floodmaps in model
                average_fps = np.min([average_fps, 1000])
                # adjust fps to floodmaps in model
                rts = np.array([rt for rt in self.model.data.inundation_maps_hist.keys()])
                idx_rp = np.argmin(abs(average_fps - rts))
                closest_fps  = rts[idx_rp]
                if closest_fps > average_fps: # make sure not to overestimate fps
                    average_fps = rts[idx_rp+1]
                else: average_fps = closest_fps


            else:
                average_fps = 2
            self.coastal_fps = average_fps  # assume a fps of 10 where there is no flopros
            self.initial_fps = int(average_fps)
        else:
            self.coastal_fps = self.model.settings['flood_risk_calculations']['flood_protection_standard']
            self.initial_fps = self.model.settings['flood_risk_calculations']['flood_protection_standard']

    def _initiate_admin_cell_coords(self):
        '''This function transforms the admin indices to lon lat for sampling data using all cells in admin geom.'''
        admin_indices = self.geom['properties']['region']['indices']

        # Get coords associated with pixels in admin region
        px = admin_indices[1][:] + 0.5
        py = admin_indices[0][:] + 0.5

        locations_admin_cells = pixel_to_coord(
            px=px, py=py, gt=self.geom['properties']['region']['gt'])
        self.locations_admin_cells = np.stack(
            [locations_admin_cells[0], locations_admin_cells[1]], axis=1)

    def _initiate_admin_cell_dist_to_coast(self):
        '''This function samples the distance to the coastline for each cell in the admin geom.'''
        # store dist_to_coast cells to instance
        self.dist_to_coast_admin_cells = self.model.data.distance_to_coast.sample_coords(
            self.locations_admin_cells).astype(np.float16)

    def _initiate_beach_proximity_bool_admin(self):
        '''This function creates a boolean for the presence of a sandy beach in an admin geom cell'''
        self.beach_proximity_bool_admin_cells = self.model.data.sandy_beach_cells.sample_coords(
            self.locations_admin_cells) == 1 * 1

    def _initiate_water_level_polynomials_admin(self, load_from_cache=True):
        '''This function estimates the polynomials and interpolates inundation depths for each admin cell and rt'''
        self.water_levels_admin_cells = FloodRisk.interpolate_water_levels(
            low_memory_mode = self.model.args.low_memory_mode,
            low_memory_mode_folder = self.model.args.low_memory_mode_folder,
            cache_file=self.cache_file,
            rcp = self.model.args.rcp,
            region=self.geom_id,
            start_year=self.model.config['general']['start_time'].year,
            end_year=self.model.config['general']['end_time'].year,
            locations=self.locations_admin_cells,         
            return_periods=np.array(
                [key for key in self.model.data.inundation_maps_hist.keys()]),
            inundation_maps={
                'hist': self.model.data.inundation_maps_hist,
                2030: self.model.data.inundation_maps_2030,
                2080: self.model.data.inundation_maps_2080},
            fps=self.coastal_fps,
            cells_on_coastline = self.cells_on_coastline,
            load_from_cache=load_from_cache
            )
    
    def _initiate_dike_lenghts(self):    
        # get cells adjecent to coast
        coastal_dike_length = self.model.data.coastal_dike_lengths.sample_coords(
            self.locations_admin_cells).astype(np.float16)
        self.cells_on_coastline = np.where(coastal_dike_length>0)[0] # check why burned in value is 255 instead of 1 in find_agents_loc... 
        self.coastal_dike_length = coastal_dike_length[self.cells_on_coastline] 

    def _initiate_admin_coastal_dikes(self):
        # get dike elevation cost
        if self.geom_id[:3] in self.model.data.dike_elevation_cost.index:
            self.dike_elevation_cost = self.model.data.dike_elevation_cost.loc[self.geom_id[:3]]['fixed_cost_scaled']
        else:
            self.model.logger.info(f'No dike_maintenance_cost cost found for {self.geom_id}')
            self.dike_elevation_cost = self.model.data.dike_elevation_cost['fixed_cost_scaled'].mean()

        # get dike maintenance cost
        if self.geom_id[:3] in self.model.data.dike_elevation_cost.index:
            self.dike_maintenance_cost = self.model.data.dike_maintenance_cost.loc[self.geom_id[:3]]['fixed_cost_scaled']
        else:
            self.model.logger.info(f'No dike_maintenance_cost cost found for {self.geom_id}')
            self.dike_maintenance_cost = self.model.data.dike_maintenance_cost['fixed_cost_scaled'].mean()

        # get all gadm admin indicator cells in geom (gadm representing government idx for decisions)
        data = self.model.data.government_gadm.sample_coords(self.locations_admin_cells)
        self.gov_admin_idx_cells = data
        self.gov_admin_idx_dikes = data[self.cells_on_coastline]
        # get fps per gov admin
        data = self.model.data.flopros.sample_coords(
            self.locations_admin_cells).astype(np.float16)
        
        self.coastal_fps_dikes = np.full_like(self.cells_on_coastline, 10)
        self.coastal_fps_gov  = {}
        self.dike_heights = np.full(self.cells_on_coastline.size, 0.0, np.float32)
        self.dikes_idx_gov = {}
        self.dike_maintenance_costs = {}
        # iterate over unique gadm regions within coastal node
        for gov_idx in np.unique(self.gov_admin_idx_cells):
            if gov_idx == -1:
                self.model.logger.info(f'No government geom found in {self.geom_id}')

            # get idx of dike cells within the government idx
            idx = np.where(self.gov_admin_idx_dikes == gov_idx)[0]
            
            # store
            self.dikes_idx_gov[gov_idx] = idx

            # check if the gadm region is coastal and has a dike
            if idx.size > 0:
                fps = data[self.cells_on_coastline[idx]]
                fps = fps[fps!=-1]
                if fps.size > 0:
                    fps_gov_admin = np.max([np.median(fps[fps!=-1]), 2])
                    if fps_gov_admin < 2: fps_gov_admin = 2
                    # adjust fps to floodmaps in model
                    rts = np.array([rt for rt in self.model.data.inundation_maps_hist.keys()])
                    idx_rp = np.argmin(abs(fps_gov_admin - rts))
                    closest_fps  = rts[idx_rp]
                    if closest_fps > fps_gov_admin: # make sure not to overestimate fps
                        fps_gov_admin = rts[idx_rp+1]
                        idx_rp += 1
                    else: fps_gov_admin = closest_fps
                    
                    self.coastal_fps_dikes[idx] = fps_gov_admin
                    assert fps_gov_admin == rts[idx_rp]
                    self.dike_heights[idx] = self.water_levels_admin_cells[idx_rp, self.cells_on_coastline[idx], 0]
                    self.dike_maintenance_costs[gov_idx] = np.sum(self.dike_heights[idx]>0) * self.dike_maintenance_cost
                    self.coastal_fps_gov[gov_idx] = fps_gov_admin    
                else:
                    fps_gov_admin = 2 # 2 is default fps when FLOPROS is missing.
                    self.coastal_fps_dikes[idx] = fps_gov_admin
                    self.coastal_fps_gov[gov_idx] = fps_gov_admin # 10 is default fps when FLOPROS is missing. 
                    self.dike_maintenance_costs[gov_idx] = 0

            # if not, just use the fps from flopros and do not assign dike height and dike fps
            else:
                # adjust fps to floodmaps in model
                idx_cells = np.where(self.gov_admin_idx_cells == gov_idx)[0]
                fps = data[idx_cells]
                fps = fps[fps != -1]
                if fps.size > 0 and np.max(fps) >= 2:
                    fps_gov_admin = np.max([np.median(fps[fps!=-1]), 2])
                    rts = np.array([rt for rt in self.model.data.inundation_maps_hist.keys()])
                    idx_rp = np.argmin(abs(fps_gov_admin - rts))
                    closest_fps  = rts[idx_rp]
                    if closest_fps > fps_gov_admin: # make sure not to overestimate fps
                        fps_gov_admin = rts[idx_rp+1]
                    else: fps_gov_admin = closest_fps
                else:
                    fps_gov_admin = 2 # 10 is default fps when FLOPROS is missing. 
                
                self.coastal_fps_gov[gov_idx] = fps_gov_admin


    def _initiate_utility_surface(self):
        '''THis function initiates a utility surface. NEEDS TO BE IMPROVED'''
        # always initialy with rcp4p5 for spinup 
        if self.model.args.rcp == 'control':
            timestep = 0
        else:
            timestep = self.model.current_time.year - self.model.config['general']['start_time'].year
        
        ead_array_admin_cells = FloodRisk.calculate_ead_cells_v2(
            coastal_fps_gov=self.coastal_fps_gov,
            gov_admin_idx_cells=self.gov_admin_idx_cells,
            n_agents=self.locations_admin_cells.shape[0],
            water_level=self.water_levels_admin_cells,
            dam_func=self.model.data.dam_func[self.UN_region_id],
            dam_func_dryproof_1m=self.model.data.dam_func_dryproof_1m[self.UN_region_id],
            property_value=self.property_value_node,
            return_periods=np.array([key for key in self.model.data.inundation_maps_hist.keys()]),
            coastal_fps=self.coastal_fps,
            timestep=timestep,
            rcp=self.model.args.rcp,
        )        

        # # Adjust for min and max distance and apply amenity function
        dist = np.array(
            self.model.data.coastal_amenity_functions['dist2coast'].index)

        amenity_factor = np.array(
            self.model.data.coastal_amenity_functions['dist2coast']['premium'])

        calc_amenity = interpolate.interp1d(x=dist, y=amenity_factor)
        
        cap_dist_to_coast_admin_cells = np.maximum(
            min(dist), self.dist_to_coast_admin_cells)

        cap_dist_to_coast_admin_cells = np.minimum(
            max(dist), self.dist_to_coast_admin_cells)

        # # Calculate utility of all cells
        if self.n != 0:
            self.coastal_amenity_premium_cells = calc_amenity(
                cap_dist_to_coast_admin_cells)            
            assert self.coastal_amenity_premium_cells.max() < 1
            self.damages_coastal_cells = ead_array_admin_cells
            self.damage_factor_cells = ead_array_admin_cells/ self.property_value_node
        
        else:
            self.damages_coastal_cells = ead_array_admin_cells * 0
            self.damage_factor_cells = ead_array_admin_cells/ self.property_value_node

        # Extract urban classes for urban mask
        urban_classes = self.model.data.SMOD.sample_coords(
            self.locations_admin_cells)

        # Urban mask
        smod_class = 13  # “Rural cluster grid cell”, if the cell belongs to a Rural Cluster spatial entity;
        # all urban intenitities including and above the rural urban claster
        # class
        self.smod_mask = np.where(urban_classes >= smod_class)[0]

        # load built up area in cells
        self.built_up_area = self.model.data.BUILT.sample_coords(
            self.locations_admin_cells)

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
        # max value of uint32, consider replacing with uint64
        assert self.max_n < 4294967295

        self._locations = np.full((self.max_n, 2), np.nan, dtype=np.float32)
        self.locations = household_locations
   

    @staticmethod
    @njit
    def _generate_household_id_per_person(
            people_indices_per_household: np.ndarray,
            n_people: int) -> np.ndarray:
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

    def _initiate_household_attribute_arrays(self):
        '''Initiates arrays for storing household attributes. It creates arrays of a fixed lenght (self.max_n) and fills them with -1.
         This is done to avoid memory reallocation during the simulation. The arrays are filled with the actual values during the simulation.
        
        '''
        # initiate empty array for household sizes and types
        self._size = np.full(self.max_n, -1, dtype=np.int8)
        self._hh_type = np.full(self.max_n, -1, dtype=np.int8)

        # initiate empty array for storing people indices per household
        # create memmap (values for inundation levels are estimated for entire run thus array do not have to be overwritten)
        if self.model.args.low_memory_mode:
            folder_name = os.path.dirname(self.cache_file)
            filename = os.path.join(folder_name, f'{self.geom_id}_people_indices_per_household.dat')
            _people_indices_per_household = np.memmap(filename, dtype='int32', mode='w+', shape=(self.max_n, self.max_household_size))
            _people_indices_per_household[:] = -1
            _people_indices_per_household.flush()
            self._people_indices_per_household = _people_indices_per_household
        else:
            self._people_indices_per_household = np.full(
                (self.max_n, self.max_household_size), -1, dtype=np.int32)
        
        # initate empty arrays for storing expected annual damages per agent
        self._ead = np.full(self.max_n, -1, np.float32)
        self._ead_dryproof = np.full(self.max_n, -1, dtype=np.float32)
        
        # iniate array for storing adaptation status and time since adaptation decision and flood experience
        self._adapt = np.full(self.max_n, -1, dtype=np.int8)
        self._time_adapt = np.full(self.max_n, -1, dtype=np.int16)
        self._flood_timer = np.full(self.max_n, -1, dtype=np.int16)
        
        # initiate array for storing would have moved
        self._would_have_moved = np.full(self.max_n, -1, dtype=np.int8)

        # initiate adaptation costs
        self._adaptation_costs = np.full(self.max_n, -1, dtype=np.int32)

        # initate array for storing flood status and count
        self._flooded = np.full(self.max_n, -1, dtype=np.int8)

        # initiate arrays for storing agent property value, income position, income, weatlh and amenity values
        self._property_value = np.full(self.max_n, -1, dtype=np.int32)
        self._income_percentile = np.full(self.max_n, -1, dtype=np.int16)
        self._income = np.full(self.max_n, -1, dtype=np.int32)
        self._wealth = np.full(self.max_n, -1, dtype=np.int32)
        self._amenity_value = np.full(self.max_n, -1, dtype=np.int32)
        self._beach_amenity = np.full(self.max_n, -1, dtype=np.int32)

        # initiate arrays with decision parameter EU for each agent
        self._decision_horizon = np.full(self.max_n, -1, dtype=np.int8)
        self._risk_aversion = np.full(self.max_n, -1, dtype=np.float32)
        self._risk_perception = np.full(self.max_n, -1, dtype=np.float32)

        # initiate array for storing agent proximity to a beach
        self._beach_proximity_bool = np.full(self.max_n, -1, dtype=np.int8)
        
        # initiate array for storing shoreline retreat experienced by agent
        self._shoreline_change_agent = np.full(
            self.max_n, -1, dtype=np.float16)
        
        # initiate array with distance to coast for each agent
        self._distance_to_coast = np.full(
            self.max_n, -1, dtype=np.float32)
        
        # initiate indice cell within admin cells for each agent
        self._indice_cell_agent = np.full(
            self.max_n, -1, dtype=np.int32)   

    def _load_damage_data_node(self):
        if self.geom_id[:3] in self.model.data.max_damage_data.index:
            self.max_damage_industrial = self.model.data.max_damage_data.loc[self.geom_id[:3]]['max_damage_industrial_LU']
            self.max_damage_commercial = self.model.data.max_damage_data.loc[self.geom_id[:3]]['max_damage_commercial_LU']
            self.max_damage_residential = self.model.data.max_damage_data.loc[self.geom_id[:3]]['max_damage_residential_LU']

        else:
            print(f'No max damage data found for node {self.geom_id}. Using regional average.')
            subset_region = self.model.data.max_damage_data[self.model.data.max_damage_data['region'] == self.UN_region_id.lower()]
            self.max_damage_industrial = subset_region['max_damage_industrial_LU'].mean()
            self.max_damage_commercial = subset_region['max_damage_commercial_LU'].mean()
            self.max_damage_residential = subset_region['max_damage_residential_LU'].mean()

    def _fill_household_attribute_arrays(self):
        '''This function fills the arrays for storing household attributes with the actual values. It is called at the end of the initate_household_attributes function.'''
        # read and fill household sizes
        size_fn = os.path.join(self.init_folder, "size.npy")
        if os.path.exists(size_fn):
            self.size = np.load(size_fn)
        
        # read and fill household types
        hh_type_fn = os.path.join(self.init_folder, "household_types.npy")
        if os.path.exists(hh_type_fn):
            self.hh_type = np.load(hh_type_fn)

        # get fixed migration costs
        if self.geom_id[:3] in self.model.data.scaled_fixed_migration_cost.index:
            self.fixed_migration_cost = self.model.data.scaled_fixed_migration_cost.loc[self.geom_id[:3]]['fixed_cost_scaled']
        else:
            # self.model.logger.info(f'No migration cost found for {self.geom_id}')
            self.fixed_migration_cost = self.model.data.scaled_fixed_migration_cost['fixed_cost_scaled'].mean()

        # get adaptation costs
        if self.geom_id[:3] in self.model.data.adaptation_cost.index:
            self.adaptation_cost = self.model.data.adaptation_cost.loc[self.geom_id[:3]]['fixed_cost_scaled']
        else:
            # self.model.logger.info(f'No adaptation cost found for {self.geom_id}')
            self.adaptation_cost = self.model.data.adaptation_cost['fixed_cost_scaled'].mean()

        # read fill household income based on income percentiles (if no data found then random income)
        income_fn =  os.path.join(self.init_folder, "household_incomes.npy")
        if os.path.exists(income_fn):
            self.income_percentile = np.load(income_fn)
            self.income_percentile = np.maximum(self.income_percentile, 1) # fix this in prepare agent data. For now this works.
        else:
            self.income_percentile = self.model.random_module.random_state.integers(1, 100, self.n)
            print(f'no household income percentiles found for {self.geom_id}')

        self.income = np.percentile(
            self.income_distribution_region,
            self.income_percentile).astype(np.int32)

        # fill arrays with property value, income
        if self.geom_id[:3] in self.model.data.max_damage_data.index:
            max_damage = self.model.data.max_damage_data.loc[self.geom_id[:3]]['max_damage_residential_object']
            if max_damage == 0:
                # self.model.logger.info(f'max damage of 0 found in {self.geom_id[:3]}')            
                max_damage = self.model.data.max_damage_data['max_damage_residential_object'].mean()

        else:
            # self.model.logger.info(f'no max damage in {self.geom_id[:3]}')            
            max_damage = self.model.data.max_damage_data['max_damage_residential_object'].mean()

        self.property_value_node = max_damage
        self.property_value = max_damage


        # fill income array
        perc = np.array([0, 20, 40, 60, 80, 100]) # percentiles
        ratio = np.array([0, 1.06, 4.14, 4.19, 5.24, 6])# wealth to income 
        self.income_wealth_ratio = interpolate.interp1d(perc, ratio) # interpolater object to assign wealth based on position in income distribution

        # assign wealth based on  position in distribution
        self.wealth = self.income_wealth_ratio(
            self.income_percentile) * self.income
        
        # self.wealth[self.wealth <
        #             self.property_value] = self.property_value[self.wealth < self.property_value]
        self.property_value = np.minimum(self.wealth, self.property_value)

        # read and fill people indices per household
        people_indices_per_household_fn = os.path.join(
            self.init_folder, "people_indices.npy")
        if os.path.exists(people_indices_per_household_fn):
            self.people_indices_per_household = np.load(
                people_indices_per_household_fn)

        # fill array storing adaptation status and time since adaptation decision and flood experience
        self.time_adapt = 0
        self.adapt = 0
        self.flooded = 0
        self.flood_timer = 99  # Assure new households have min risk perceptions

        # fill adaptation costs
        total_cost = self.adaptation_cost
        loan_duration = self.model.settings['adaptation']['loan_duration']
        r_loan = self.model.settings['adaptation']['interest_rate']
        self.annual_adaptation_cost = total_cost * \
            (r_loan * (1 + r_loan) ** loan_duration / ((1 + r_loan)**loan_duration - 1))
        self.adaptation_costs = self.annual_adaptation_cost

        # fill arrays with decision parameter EU for each agent
        self.decision_horizon = self.model.settings['decisions']['decision_horizon']
        self.risk_aversion = self.model.settings['decisions']['risk_aversion']
        self.risk_perception = self.model.settings['flood_risk_calculations']['risk_perception']['min']
        
        # fill shoreline retreat tracker admin and agent
        self.total_shoreline_change_admin = 0
        self.shoreline_change_agent = 0

        # fill array for storing agent amenity value
        self.average_amenity_value = 0

        # fill indice agent cell
        agent_indices = coords_to_pixels(self.locations, self.geom['properties']['region']['gt'])       
        region_indices = self.geom['properties']['region']['indices']

        self.indice_cell_agent = self.get_agent_indice_admin_cells(
            agent_indices=agent_indices,
            region_indices=region_indices
            )
        
        assert all(self.indice_cell_agent != -1)
        self._initiate_utility_surface()

        # get beach proximity and distance to coast for agents by sampling from admin geom cells
        self.beach_proximity_bool = self.beach_proximity_bool_admin_cells[self.indice_cell_agent]
        self.distance_to_coast = self.dist_to_coast_admin_cells[self.indice_cell_agent]
      
        # set household status for would have moved (if adaptation was not an option).
        self.would_have_moved = 0


    def _initiate_household_attributes(self):
        '''This function initiates and fills the arrays storing household attributes'''
        self._initiate_household_attribute_arrays()
        self._fill_household_attribute_arrays()

        # calculate population size in coastal node
        self.population = np.sum(self.size)

    def _initiate_person_memmap(self):
        '''The function creates a memory mapped array to store person attributes.
        The person attributes age and gender are manipulated through getter and setter functions.'''
        # initiate empty memmap array to store person attributes
        n_people = np.count_nonzero(self._people_indices_per_household != -1)
        n_person_attributes = 2 # only gender and age are person attributes, else adjust
        # get folder to store memmaps
        folder_name = os.path.dirname(self.cache_file)

        # creat memmap object for _empty_index_stack_counter
        filename = os.path.join(folder_name, f'{self.geom_id}_empty_index_stack.dat')
        _empty_index_stack = np.memmap(filename, dtype='int32', mode='w+', shape=n_people + self.person_reduncancy)
        _empty_index_stack[:] = -1
        size_empty_stack = _empty_index_stack.size - n_people          
        _empty_index_stack[:size_empty_stack] = np.arange(
                    n_people, _empty_index_stack.size)[::-1]
        _empty_index_stack.flush()
        self._empty_index_stack = _empty_index_stack
        self._empty_index_stack_counter = size_empty_stack - 1

        # create memmap object for _person_attribute_array
        filename = os.path.join(folder_name, f'{self.geom_id}_person_attribute_array.dat')
        _person_attribute_array = np.memmap(filename, dtype='int8', mode='w+', shape=(n_person_attributes, self._empty_index_stack.size))
        _person_attribute_array[:] = -1
        _person_attribute_array.flush()
        self._person_attribute_array = _person_attribute_array
        
        # create memmap object for _household_id_per_person
        filename = os.path.join(folder_name, f'{self.geom_id}_household_id_per_person.dat')
        _household_id_per_person = np.memmap(filename, dtype='int32', mode='w+', shape=self.max_n_people)
        _household_id_per_person[:] = -1
        _household_id_per_person.flush()
        self._household_id_per_person = _household_id_per_person  

    def update_memmap_paths(self):
        '''This function updates the path to the memmaps of person attributes to prevent them from being overwritten after the spinup period.
        It basically creates new memmaps in the run folder.'''

        folder_name = os.path.dirname(self.cache_file)

        # recreate memmap object for _empty_index_stack_counter
        filename = os.path.join(folder_name, f'{self.geom_id}_empty_index_stack.dat')
        _empty_index_stack = np.memmap(filename, dtype='int32', mode='w+', shape=self._empty_index_stack.shape)
        _empty_index_stack[:] = self._empty_index_stack[:] # fill with current values
        _empty_index_stack.flush()
        delattr(self, '_empty_index_stack')
        self._empty_index_stack = _empty_index_stack

        # create memmap object for _person_attribute_array
        filename = os.path.join(folder_name, f'{self.geom_id}_person_attribute_array.dat')
        _person_attribute_array = np.memmap(filename, dtype='int8', mode='w+', shape=self._person_attribute_array.shape)
        _person_attribute_array[:] = self._person_attribute_array[:]
        _person_attribute_array.flush()
        delattr(self, '_person_attribute_array')
        self._person_attribute_array = _person_attribute_array
        
        # create memmap object for _household_id_per_person
        filename = os.path.join(folder_name, f'{self.geom_id}_household_id_per_person.dat')
        _household_id_per_person = np.memmap(filename, dtype='int32', mode='w+', shape=self._household_id_per_person.shape)
        _household_id_per_person[:] = self._household_id_per_person[:]
        _household_id_per_person.flush()
        delattr(self, '_household_id_per_person')
        self._household_id_per_person = _household_id_per_person  

        # recreate people indices per household
        filename = os.path.join(folder_name, f'{self.geom_id}_people_indices_per_household.dat')
        _people_indices_per_household = np.memmap(filename, dtype='int32', mode='w+', shape=self._people_indices_per_household.shape)
        _people_indices_per_household[:] = self._people_indices_per_household[:]
        _people_indices_per_household.flush()
        delattr(self, '_people_indices_per_household')
        self._people_indices_per_household = _people_indices_per_household

    def _initiate_person_attributes(self):
        '''This function initiates and fills the arrays for storing person attributes'''
        n_people = np.count_nonzero(self._people_indices_per_household != -1)
        n_person_attributes = 2 # only gender and age are person attributes, else adjust

        if self.model.args.low_memory_mode:
            self._initiate_person_memmap()
        else:  
            self._empty_index_stack = np.full(
                n_people + self.person_reduncancy, -1, dtype=np.int32)
            size_empty_stack = self._empty_index_stack.size - n_people
            self._empty_index_stack[:size_empty_stack] = np.arange(
                n_people, self._empty_index_stack.size)[::-1]
            self._empty_index_stack_counter = size_empty_stack - 1
            self._person_attribute_array = np.full((n_person_attributes, self._empty_index_stack.size), -1, dtype=np.int8)
            self._household_id_per_person = np.full(
                self.max_n_people, -1, dtype=np.int32)

        gender_fn = os.path.join(self.init_folder, "gender.npy")
        if os.path.exists(gender_fn):
            self.gender = np.load(gender_fn)

        age_fn = os.path.join(self.init_folder, "age.npy")
        if os.path.exists(age_fn):
            self.age = np.load(age_fn)

        self.household_id_per_person = self._generate_household_id_per_person(
            self._people_indices_per_household, self.size.sum())

        # assert self._gender.shape == self._risk_aversion.shape
        assert self.age.shape == self.gender.shape

        # Assert dataw
        # assert np.array_equal(self._gender == -1, self._risk_aversion == -1)
        assert np.array_equal(self.gender == -1, self.age == -1)

    def load_timestep_data(self):
        pass

    def process(self):
        self._household_id_per_person = self._generate_household_id_per_person(
            self._people_indices_per_household, self.size.sum())

    def update_utility_surface(self):
        '''This function updates the utility for cells in the coastal node'''
        # Read ead from flood risk class instance
        if self.model.args.rcp == 'control':
            timestep = 0
        else:
            timestep = self.model.current_time.year - self.model.config['general']['start_time'].year
        
        ead_array_admin_cells = FloodRisk.calculate_ead_cells_v2(
            coastal_fps_gov=self.coastal_fps_gov,
            gov_admin_idx_cells=self.gov_admin_idx_cells,
            n_agents=self.locations_admin_cells.shape[0],
            water_level=self.water_levels_admin_cells,
            dam_func=self.model.data.dam_func[self.UN_region_id],
            dam_func_dryproof_1m=self.model.data.dam_func_dryproof_1m[self.UN_region_id],
            property_value=self.property_value_node,
            return_periods=np.array([key for key in self.model.data.inundation_maps_hist.keys()]),
            coastal_fps=self.coastal_fps,
            timestep=timestep,
            rcp=self.model.args.rcp,

        )        

        # if agents are present in the coastal node
        if self.n > 0:
            
            # get admin indices relative to beach cells
            admin_indices_beach_cells = self.agents.beaches.admin_indices_beach_cells[self.geom_id]

            # rotate(?) and assign to node
            self.admin_indices_beach_cells = tuple([admin_indices_beach_cells[1], admin_indices_beach_cells[0]])

            self.damages_coastal_cells = ead_array_admin_cells
            # get beach widths in admin cells 
            beach_widths_admin = self.agents.beaches.beach_width[self.admin_indices_beach_cells]

            # calculate beach amenities cells
            beach_mask = beach_widths_admin != -1
            self.beach_amenity_premium_cells = 0# self.model.coastal_amenities.calculate_beach_amenity(np.full(beach_mask.size, 1), beach_widths_admin, beach_mask)

    def calculate_flood_risk(self):  # Change this to calculate risk
        '''This class method calls a number of functions to update flood risk related infomcation. '''
        # settings required for sensitivity analysis
        # Check if erosion module has been called already. If not, initiate with the initial beach width
        if hasattr(self, 'beach_width_floodplain'):
            beach_width_floodplain = self.beach_width_floodplain
        else:
            beach_width_floodplain = np.full(self.n,
                self.model.settings['shoreline_change']['initial_beach_width'],
                np.float32)

        erosion_effect_fps = 5
        if hasattr(self.model.args, 'sensitivity_run'):
            if self.model.args.sensitivity_run == 'include_erosion_effect':
                erosion_effect_fps = self.model.args.erosion_effect_fps              

        water_level, _, households_in_100yr_fp = FloodRisk.sample_water_level(
            admin_name=self.geom_id,
            gov_admin_idx_cells=self.gov_admin_idx_cells,
            dikes_idx_govs=self.dikes_idx_gov,
            coastal_fps_gov=self.coastal_fps_gov,
            dike_heights=self.dike_heights,
            cells_on_coastline=self.cells_on_coastline,
            water_levels_admin_cells=self.water_levels_admin_cells,
            indice_cell_agent = self.indice_cell_agent,
            return_periods=np.array([key for key in self.model.data.inundation_maps_hist.keys()]),
            fps=self.coastal_fps,
            fps_dikes=self.coastal_fps_dikes,
            strategy=self.model.args.government_strategy,
            start_year = self.model.config['general']['start_time'].year,
            current_year=self.model.current_time.year,
            rcp=self.model.args.rcp,
            beach_width_floodplain=beach_width_floodplain, 
            beach_mask=self.beach_proximity_bool == 1, 
            erosion_effect_fps=erosion_effect_fps)

        self.flooded, self.risk_perception, self.flood_timer, self.flood_tracker = FloodRisk.stochastic_flood(
            random_state=self.model.random_module.random_state_flood,
            current_geom=self.geom_id,
            water_levels=water_level,
            return_periods=np.array([key for key in self.model.data.inundation_maps_hist.keys()]),
            flooded=self.flooded,
            risk_perceptions=self.risk_perception,
            flood_timer=self.flood_timer,
            risk_perc_min=self.model.settings['flood_risk_calculations']['risk_perception']['min'],
            risk_perc_max=self.model.settings['flood_risk_calculations']['risk_perception']['max'],
            risk_decr=self.model.settings['flood_risk_calculations']['risk_perception']['coef'],
            settings=self.model.settings['general']['flood'],
            fps_dikes=self.coastal_fps_dikes,
            current_year=self.model.current_time.year,
            spin_up_flag=self.model.spin_up_flag,
            flood_tracker=self.flood_tracker)

        self.damages, self.damages_dryproof_1m, self.ead_residential = FloodRisk.calculate_ead(
            n_agents=self.n,
            adapted = self.adapt,
            water_level=water_level,
            dam_func=self.model.data.dam_func[self.UN_region_id],
            dam_func_dryproof_1m=self.model.data.dam_func_dryproof_1m[self.UN_region_id],
            property_value=self.property_value,
            return_periods=np.array([key for key in self.model.data.inundation_maps_hist.keys()]),
            coastal_fps=self.coastal_fps,
            coastal_fps_gov=self.coastal_fps_gov,
            initial_fps = self.initial_fps,
            gov_admin_idx_cells=self.gov_admin_idx_cells,
            indice_cell_agent=self.indice_cell_agent
            )
               
        if self.model.args.rcp == 'control':
            timestep = 0
        else:
            timestep = self.model.current_time.year - self.model.config['general']['start_time'].year

        ead_commercial = FloodRisk.calculate_ead_cells_LU(
            coastal_fps_gov=self.coastal_fps_gov,
            gov_admin_idx_cells=self.gov_admin_idx_cells,
            n_agents=self.locations_admin_cells.shape[0],
            water_level=self.water_levels_admin_cells,
            dam_func=self.model.data.dam_func_commercial[self.UN_region_id],
            max_dam=self.max_damage_commercial,
            return_periods=np.array([key for key in self.model.data.inundation_maps_hist.keys()]),
            area_of_grid_cell=0.15,
            built_up_area = self.built_up_area,
            timestep=timestep,
            rcp=self.model.args.rcp,
        )       

        ead_industrial = FloodRisk.calculate_ead_cells_LU(
            coastal_fps_gov=self.coastal_fps_gov,
            gov_admin_idx_cells=self.gov_admin_idx_cells,
            n_agents=self.locations_admin_cells.shape[0],
            water_level=self.water_levels_admin_cells,
            dam_func=self.model.data.dam_func_industrial[self.UN_region_id],
            max_dam=self.max_damage_industrial,
            return_periods=np.array([key for key in self.model.data.inundation_maps_hist.keys()]),
            area_of_grid_cell=0.10,
            built_up_area = self.built_up_area,
            timestep=timestep,
            rcp=self.model.args.rcp,
        )   

        ead_residential_lu = FloodRisk.calculate_ead_cells_LU(
            coastal_fps_gov=self.coastal_fps_gov,
            gov_admin_idx_cells=self.gov_admin_idx_cells,
            n_agents=self.locations_admin_cells.shape[0],
            water_level=self.water_levels_admin_cells,
            dam_func=self.model.data.dam_func[self.UN_region_id],
            max_dam=self.max_damage_residential,
            return_periods=np.array([key for key in self.model.data.inundation_maps_hist.keys()]),
            area_of_grid_cell=0.75,
            built_up_area = self.built_up_area,
            timestep=timestep,
            rcp=self.model.args.rcp,
        )

        # add to ead total
        self.ead_total = self.ead_residential + ead_commercial + ead_industrial

        # also store ead per land use
        self.ead_residential_land_use = ead_residential_lu
        self.ead_commercial_land_use = ead_commercial
        self.ead_industrial_land_use = ead_industrial

        ### DEBUGGING calculate ead ########################
        rps = np.array(
            [key for key in self.model.data.inundation_maps_hist.keys()])
        p_floods = 1/ rps
        self.ead_agents = np.trapz(self.damages, p_floods, axis = 0)
        self.n_households_exposed = np.sum(self.ead_agents > 0)
        self.ead_agents[self.adapt == 1] = np.trapz(self.damages_dryproof_1m[:, self.adapt == 1], p_floods, axis = 0)
        #####################################################

        # calculate n people in current 100yr floodplain
        self.population_in_100yr_floodplain = np.sum(self.size[households_in_100yr_fp])


    def process_coastal_erosion(self):
        '''This function processes coastal erosion. It transforms lat lon of cells located in the admin geom to indices relive to
        the large beach array. It then extracts the beach width in the admin cells and assigns these to the household agents.'''
        # if self.model.args.low_memory_mode:
        #     self.load_arrays_from_npz()

        # create mask of people in beach cell
        beach_mask = self.beach_proximity_bool == 1

        # sum people in beach cells
        self.people_near_beach = np.sum(self.size[beach_mask])
        self.households_near_beach = np.sum(self.beach_proximity_bool)

        # there could never be more people living in beach cells than in the entire admin
        assert self.people_near_beach <= self.population 

        # Transform gadm location coords to indices relative to large beach
        # array
        y, x = coords_to_pixels(
            coords=self.locations_admin_cells[self.beach_proximity_bool_admin_cells], gt=self.agents.beaches.beach_data_gt)

        self.indices_admin_in_beach_cells = tuple([x, y])      

        # preallocate array filled with beach width of 0m
        beach_width_admin_cells = np.zeros(self.beach_proximity_bool_admin_cells.size, np.float32)

        # sample from beach width array
        beach_width_admin_cells[self.beach_proximity_bool_admin_cells] = self.agents.beaches.beach_width[self.indices_admin_in_beach_cells]

        # assert only beach cells are sampled
        assert (beach_width_admin_cells[self.beach_proximity_bool_admin_cells] != -1).all()

        # assign for export to coastal manager and calculation of coastal amenity
        self.beach_width_agents = np.take(beach_width_admin_cells, self.indice_cell_agent)

    def calculate_coastal_amenity_values(self):

        # since we do not yet account for beach amenity, just collect amenity premium from cells in coastal admin
        self.coastal_amenity_premium = self.coastal_amenity_premium_cells[self.indice_cell_agent]
        self.amenity_value = self.coastal_amenity_premium * self.wealth
        
        # process coastal erosion
        # self.process_coastal_erosion()

        # # Calculate amenities based on beach proximity and distance to coast
        # beach_mask = self.beach_proximity_bool == 1
        # self.coastal_amenity_premium, self.amenity_value, beach_amenity = self.model.coastal_amenities.total_amenity(
        #     beach_proximity_bool=beach_mask,
        #     distance_to_coast = self.distance_to_coast,
        #     beach_width=self.beach_width_agents,
        #     agent_wealth=self.wealth,
        # )

        # assert beach_amenity.size == self.beach_proximity_bool.size
        # self.beach_amenity = beach_amenity

    def initiate_household_attributes_movers(self, n_movers):
        '''This function assigns new household attributes to the households that moved into a floodplain. It takes all arrays and fills in the missing data based on sampling.'''

        assert n_movers == 0 or self.income[-n_movers] == -1

        # fill indice agent cell new households
        agent_indices = coords_to_pixels(self.locations[-n_movers:], self.geom['properties']['region']['gt'])       
        region_indices = self.geom['properties']['region']['indices']

        self.indice_cell_agent[-n_movers:] = self.get_agent_indice_admin_cells(
            agent_indices=agent_indices,
            region_indices=region_indices
            )

        assert all(self.indice_cell_agent != -1)

        # Sample income percentile for households moving in from inland node or natural pop change
        # Find neighbors for newly generated households
        new_households = np.where(self.income_percentile == -99)[0]
        
        # ugly way to set household type movers
        size_movers = self.size[new_households]
        self.hh_type[new_households] = np.minimum(3, size_movers)
        assert all(self.hh_type != -99)

        # Assign income to agents based on similar households in the region
        # iterate over households and fill income percentile
        # get income percentiles per household class
        household_type_movers, counts = np.unique(self.hh_type[new_households], return_counts=True)
        
        for household_type, count in zip(household_type_movers, counts):    
            # get where to fill 
            to_fill = np.where(np.logical_and(self.income_percentile == -99, self.hh_type == household_type))
            # get sample from precentile in region (if houseshold type present in region)
            if household_type in self.hh_type[:-n_movers]:
                percentile_hh_type = self.income_percentile[:-n_movers][self.hh_type[:-n_movers] == household_type]
                income_percentile_new_households = self.model.random_module.random_state.choice(percentile_hh_type, count)
                self.income_percentile[to_fill] = income_percentile_new_households
            else:
                self.income_percentile[to_fill] = self.model.random_module.random_state.choice(self.income_percentile[:-n_movers], count)

        assert (self.income_percentile != -99).all()
        
          
        self.income[-n_movers:] = np.percentile(
            self.income_distribution_region, self.income_percentile[-n_movers:])
        assert (self.income >= 0).all()

        self.wealth[-n_movers:] = self.income[-n_movers:] * \
            self.income_wealth_ratio(self.income_percentile[-n_movers:])

        # All agents have the same property value
        self.property_value[-n_movers:] = np.minimum(self.property_value_node, self.wealth[-n_movers:])

        # # Set wealth to never be lower than property value
        # self.wealth[self.wealth <
        #             self.property_value] = self.property_value[self.wealth < self.property_value]

        # Set decision horizon and flood timer
        # self.risk_aversion[-n_movers:] = self.model.settings['decisions']['risk_aversion']
        self.decision_horizon[-n_movers:] = self.model.settings['decisions']['decision_horizon']
        self.flood_timer[-n_movers:] = 99
        self.time_adapt[-n_movers:] = 0
        self.adaptation_costs[-n_movers:] = self.annual_adaptation_cost
        
        assert (self.risk_aversion != -1).all()
        assert (self.decision_horizon != -1).all()
        assert (self.time_adapt != -1).all()
        assert (self.adaptation_costs != -1).all()
        # sample beach proximity bool for new households
        indices_new_households = np.where(self.beach_proximity_bool == -1)
        assert indices_new_households[0].size == n_movers
        # only sample beach proximity for new households
        self.beach_proximity_bool[indices_new_households] = self.beach_proximity_bool_admin_cells[self.indice_cell_agent[indices_new_households]]

        # Transform household location coords to indices relative to beach
        # array      
        
        # sample distance to coast for movers
        self.distance_to_coast[-n_movers:] = self.dist_to_coast_admin_cells[self.indice_cell_agent[-n_movers:] ]
        assert all(self.distance_to_coast != -1)

        # Reset flood status to 0 for all households (old and new) and
        # adaptation to 0 for new households
        self.adapt[-n_movers:] = 0

    def process_population_change(self):
        '''This function processes population change in the coastal node. It is called at the beginning of each time step. 
        It samples the number of households to remove and adds new households based on the population change. 
        It also updates the population attribute of the coastal node.'''

        population_change, household_sizes = self.ambient_pop_change()

        households_to_remove = []
        if population_change < 0 and self.population > abs(population_change):
            # Select households to remove from
            # Sample the households that are due for removal from self.size
            individuals_removed = 0
            households_to_remove = []
            i = 0

            # iterate while number of individuals removed does not meet
            # projections or iterations exceed limit
            while individuals_removed < abs(population_change) and i < 1E6:
                household = self.model.random_module.random_state.integers(0, self.size.size)
                if household not in households_to_remove:
                    individuals_removed += self.size[household]
                    households_to_remove.append(household)
                i += 1
            households_to_remove = np.sort(households_to_remove)[::-1]

            n_movers = np.sum(self.size[households_to_remove])
            # Placeholder, will not do anythin with this
            move_to_region = np.full(n_movers, self.admin_idx, dtype=np.int16)

            # Remove households from abm
            self.population, self.n, self._empty_index_stack_counter, _, _, _, _, _, _, _, _, _, _ = self.move_numba(
                population=self.population,
                n=self.n,
                people_indices_per_household=self._people_indices_per_household,
                empty_index_stack=self._empty_index_stack,
                empty_index_stack_counter=self._empty_index_stack_counter,
                indice_cell_agent=self.indice_cell_agent,
                households_to_move=households_to_remove,
                n_movers=n_movers,
                move_to_region=move_to_region,
                admin_idx=self.admin_idx,
                locations=self._locations,
                size=self._size,
                hh_type=self._hh_type,
                ead=self._ead,
                ead_dryproof=self._ead_dryproof,
                gender=self._person_attribute_array[0, :],
                age=self._person_attribute_array[1, :],
                risk_aversion=self._risk_aversion,
                income_percentile=self._income_percentile,
                income=self._income,
                wealth=self._wealth,
                risk_perception=self._risk_perception,
                flood_timer=self._flood_timer,
                adapt=self._adapt,
                would_have_moved=self._would_have_moved,
                adaptation_costs=self._adaptation_costs,
                time_adapt=self._time_adapt,
                decision_horizon=self._decision_horizon,
                property_value=self._property_value,
                amenity_value=self._amenity_value,
                beach_proximity_bool=self._beach_proximity_bool,
                beach_amenity=self._beach_amenity,
                distance_to_coast=self._distance_to_coast
            )

            # and update mask
            self.create_mask_household_allocation()


        elif population_change > 0:
            # Generate attributes
            n_movers, to_region, household_id, gender, age, income_percentile, household_type, income, risk_perception, risk_aversion  = self._generate_households(
                n_households_to_move=household_sizes.size,
                household_sizes=household_sizes,
                move_to_region_per_household=np.full(
                    household_sizes.size,
                    self.admin_idx),
                # new households as 'moved' to own region
                init_risk_aversion=self.model.settings['decisions']['risk_aversion'],
                init_risk_perception=self.model.settings['flood_risk_calculations']['risk_perception']['min'],
            )
            # create array with cells to move to (only relevant for moving to coastal node)
            cells_to_move_to = np.full(household_sizes.size, -1, np.int64)


            people = {
                "from": np.full(
                    n_movers,
                    self.admin_idx,
                    dtype=np.int16),
                "to": to_region,
                "household_id": household_id,
                "gender": gender,
                "age": age,

            }
            households = {
                "from": np.full(
                    n_movers,
                    self.admin_idx,
                    dtype=np.int16),
                "to": to_region[household_id],
                "household_id": np.unique(household_id),
                "risk_aversion": risk_aversion,
                "income_percentile": income_percentile,
                "household_type": household_type,
                "income": income,
                "risk_perception": risk_perception,
                "cells_to_move_to": cells_to_move_to
               }

            # Add households
            self.add(people, households, pop_change = True)
        elif population_change == 0:
            pass
        


    def step(self):
        # if in first timestep after spinup and running from cache change export folder
        if not self.model.spin_up_flag and self.model.current_time.year == 2016 and self.model.args.run_from_cache:
            self.update_cache_path()
            # again sample water levels (consistant with rcp)
            self._initiate_water_level_polynomials_admin()
            self.update_memmap_paths()

        if self.model.settings['general']['include_ambient_pop_change']:
            self.process_population_change()
        
        self.load_timestep_data()
        self.process()

        if self.model.config['general']['create_agents']:
            self.calculate_amenity_values_available_cells()
            self.calculate_damage_available_cells()
            self.calculate_coastal_amenity_values()
            self.calculate_flood_risk()
            self.update_utility_surface()
            # sum annual adaptation costs incurred by agents
            self.annual_summed_adaptation_costs = self.annual_adaptation_cost * self.adapt[self.time_adapt < self.model.settings['adaptation']['loan_duration']].sum()


    @staticmethod
    @njit
    def add_numba(
        max_household_density,
        n_households_in_grid_cell,
        admin_name: str,
        n: int,
        nr_cells_to_assess,
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
        new_risk_aversions: np.ndarray,
        new_income_percentiles: np.ndarray,
        new_household_types: np.ndarray,
        new_risk_perceptions: np.ndarray,
        gender_movers: np.ndarray,
        age_movers: np.ndarray,
        locations: np.ndarray,
        size: np.ndarray,
        gender: np.ndarray,
        age: np.ndarray,
        
        cells_to_move_to,
        risk_aversion: np.ndarray,
        would_have_moved: np.ndarray,
        income_percentile: np.ndarray,
        household_type: np.ndarray,
        risk_perception: np.ndarray,
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

        # a priori generate random numbers (test)
        redundancy = new_household_sizes.size # generate a frame with as size the number of households moving in.
        sample_size = np.minimum(nr_cells_to_assess, smod_mask.size)
        assert smod_mask.size > 0

        cells_to_assess_array = np.full(
            (redundancy, sample_size), -1, dtype=np.int64)
        for i in range(redundancy):
            cells_to_assess_array[i, :] = np.random.choice(
                smod_mask, size=sample_size, replace=False)       

        adjust_y = np.random.random(redundancy)
        adjust_x = np.random.random(redundancy)
        index = 0

        for index_first_person_in_household, new_household_size, new_income_percentile, new_household_type, new_risk_perception, new_risk_aversion, cell_to_move_to in zip(
                index_first_persons_in_household, new_household_sizes, new_income_percentiles, new_household_types, new_risk_perceptions, new_risk_aversions, cells_to_move_to):
            n += 1
            assert n > 0
            assert size[n - 1] == -1
            assert income_percentile[n - 1] == -1

            size[n - 1] = new_household_size
            risk_aversion[n - 1] = new_risk_aversion
            income_percentile[n - 1] = new_income_percentile
            household_type[n-1] = new_household_type
            risk_perception[n - 1] = new_risk_perception
            would_have_moved[n -1] = 0

            # Fill individual attributes
            for i in range(new_household_size):
                # get an empty index
                empty_index = empty_index_stack[empty_index_stack_counter]
                # check if we don't get an empty index
                assert empty_index != -1
                # check if spot is empty
                assert gender[empty_index] == -1
                assert age[empty_index] == -1
                # set gender, risk aversion and age
                gender[empty_index] = gender_movers[index_first_person_in_household + i]
                age[empty_index] = age_movers[index_first_person_in_household + i]
                # emtpy the index stack
                empty_index_stack[empty_index_stack_counter] = -1
                # and finally decrement stack counter
                empty_index_stack_counter -= 1
                if empty_index_stack_counter < 0:
                    raise OverflowError(
                        "Too many agents in class. Consider increasing redundancy")
                assert empty_index_stack[empty_index_stack_counter] != -1
                # set empty index in people indices
                people_indices_per_household[n - 1, i] = empty_index
       
            # check if the destination cell was not yet selected (happens in moves inlanc -> coastal and in pop change)
            if cell_to_move_to == -1: 
                # Select sample of random cells:
                cells_to_assess = cells_to_assess_array[index, :]
                
                # calculate avaiable housing in grid cells
                n_housing_left_in_grid_cell = max_household_density - n_households_in_grid_cell
                
                # get n housing left in cell
                n_housing_left_in_grid_cell[cells_to_assess]

                # get cells that are still at max household density
                n_available_spots_in_selected_cells = n_housing_left_in_grid_cell[cells_to_assess]
                unavailable_cells = np.where(n_available_spots_in_selected_cells <= 0)
                
                if unavailable_cells[0].size == cells_to_assess.size:
                    # print('Warning, selected cells are full. Now considering all SMOD cells')
                    cells_to_assess = smod_mask
                    # get n housing left in cell
                    n_housing_left_in_grid_cell[cells_to_assess]
                    # get cells that are still at max household density
                    n_available_spots_in_selected_cells = n_housing_left_in_grid_cell[cells_to_assess]
                    unavailable_cells = np.where(n_available_spots_in_selected_cells <= 0)
                    if unavailable_cells[0].size == cells_to_assess.size:
                        # print(f'[add_numba] warning, cells full in {admin_name}, now allocating at best cell of {damages_coastal_cells.size} cells in region.')
                        cells_to_assess = np.arange(damages_coastal_cells.size) # Select random cells

                # select cells to assess by the agent
                damages_coastal_cells_agent = damages_coastal_cells[cells_to_assess]
                coastal_amenity_cells_agent = coastal_amenity_cells[cells_to_assess]

                # multiply expected damages with risk perception of the households
                # account for risk perceptions of -1, will be set later (household
                # not moving in from other coastal nodes)
                agent_risk_perception = np.maximum(0.01, new_risk_perception)
                damages_coastal_cells_agent = agent_risk_perception * damages_coastal_cells_agent

                # multiply amenity value with amenity weight
                coastal_amenity_cells_agent = coastal_amenity_cells_agent * amenity_weight

                # calculate utility cells
                utility_cells = coastal_amenity_cells_agent - damages_coastal_cells_agent

                # set utility for full cells to zero
                utility_cells[unavailable_cells] = -np.inf

                # pick cell with highest utility
                cell_idx = np.argsort(utility_cells)[-1:]
                cell = cells_to_assess[cell_idx]  # array should be unpacked for njit
            
            else:
                # if the destination cell was already determined use this one (moves from coastal -> coastal)
                cell = np.array([cell_to_move_to]) # numba wants array (?)
            # add household to n households in cell
            n_households_in_grid_cell[cell] += 1

            # Allocate randomly in 1km2 grid cell (otherwise all agents the
            # center of the cell) # Are actually positioned at the top left
            # corner
            px = admin_indices[1][cell] + adjust_x[index]
            py = admin_indices[0][cell] + adjust_y[index]

            locations[n - 1] = np.concatenate(pixel_to_coord(px, py, gt))
            index += 1
        return n, empty_index_stack_counter

    def _initiate_max_household_density_admin(self):
        '''Initiate the maximum household density for the admin unit. 
        This is the maximum number of households that can be in a grid cell. 
        If the number of households in a cell exceeds this number, no more households can move in. 
        This is used to limit the growth of the admin unit.'''

        # get n households in grid cells and set max household density
        if self.model.settings['decisions']['migration']['limit_admin_growth']:
            _, n_households_in_cell = np.unique(self.indice_cell_agent, return_counts=True)
            self.max_household_density = n_households_in_cell.max()
        else:
            self.max_household_density = np.inf

    def calculate_amenity_values_available_cells(self):
        if hasattr(self, 'beach_amenity_premium_cells'):
            coastal_amenity_premium_cells = self.coastal_amenity_premium_cells + self.beach_amenity_premium_cells 
        else:
            coastal_amenity_premium_cells = self.coastal_amenity_premium_cells
        self.amenity_premium_available_cells = coastal_amenity_premium_cells# [self.mask_household_allocation]

    def calculate_damage_available_cells(self):
        damage_factor_coastal_cells = self.damages_coastal_cells/ self.property_value_node
        self.damage_factor_cells = damage_factor_coastal_cells# [self.mask_household_allocation]

    def create_mask_household_allocation(self):
        # get n households in grid cells
        n_household_in_grid_cell = np.full(self.locations_admin_cells.shape[0], 0)
        populated_cells, n_households_in_cell = np.unique(self.indice_cell_agent, return_counts=True)
        # assert n_households_in_cell.max() <= self.max_household_density
        # fill array with population in cells
        n_household_in_grid_cell[populated_cells] = n_households_in_cell
        self.n_household_in_grid_cell = n_household_in_grid_cell
        # update mask to only select cells where growth is allowed (pop dens less than max initial pop density)
        mask_pop = np.where(np.logical_and(self.n_household_in_grid_cell <= self.max_household_density, self.n_household_in_grid_cell > 0))[0] # only allow population growth in already populated cells
        mask_smod = self.smod_mask
        # get common elements if some urban area in location, else use populated grid cells as mask
        if self.smod_mask.size > 5:
            self.mask_household_allocation = np.intersect1d(mask_pop, mask_smod)  
            if n_household_in_grid_cell[self.mask_household_allocation].max() > self.max_household_density:
                self.model.logger.info(f'{n_household_in_grid_cell[self.mask_household_allocation].max()} households where {self.max_household_density} is max density in {self.geom_id}')
            # assert n_household_in_grid_cell[self.mask_household_allocation].max() <= self.max_household_density
        else:
            self.mask_household_allocation = mask_pop
            if n_household_in_grid_cell[self.mask_household_allocation].size > 0:
                if n_household_in_grid_cell[self.mask_household_allocation].max() > self.max_household_density:
                    self.model.logger.info(f'{n_household_in_grid_cell[self.mask_household_allocation].max()} households where {self.max_household_density} is max density in {self.geom_id}')

        # calculate n_available housing    
        self.n_available_housing = np.sum(self.max_household_density - n_household_in_grid_cell[self.mask_household_allocation])
 

        # set status
        # set status to full if less than a percentage of housing in cell is still available
        max_growth = self.model.settings['decisions']['migration']['max_admin_growth']
        if self.n_available_housing > (max_growth * self.n):
            self.admin_full = False
        else:
            self.admin_full = True
            self.mask_household_allocation = populated_cells
            self.model.logger.info(f'{self.geom_id} full.')

        self.mask_household_allocation = self.mask_household_allocation.astype(np.int64)
        assert self.mask_household_allocation.size > 0


    def add(self, people: dict[np.ndarray], households: dict[np.ndarray], pop_change = False) -> None:
        """This function adds new households to a region. As input the function takes a dictionary of people characteristics, such as age and gender. In addition, the key and corresponding array 'household_id' is used to determine the household that the person belongs to."""
        index_first_persons_in_household, new_household_sizes = np.unique(
            people['household_id'], return_index=True, return_counts=True)[1:]

        # Extract household income percentile from people array
        new_income_percentiles = households['income_percentile']
        new_household_types = households['household_type']
        new_risk_perceptions = households['risk_perception']
        new_risk_aversions = households['risk_aversion']
        new_risk_aversions = households['risk_aversion']
        new_risk_aversions = households['risk_aversion']       
        cells_to_move_to = households['cells_to_move_to']

        self.n_moved_in_last_timestep = index_first_persons_in_household.size

        if self.model.settings['decisions']['migration']['account_for_coastal_amenity_in_allocation']:
            coastal_amenity_cells = self.coastal_amenity_premium_cells  * self.wealth.mean() * self.model.settings['decisions']['migration']['amenity_weight']
        else:
            coastal_amenity_cells = np.zeros_like(self.coastal_amenity_premium_cells)

        self.n, self._empty_index_stack_counter = self.add_numba(
            max_household_density = self.max_household_density,
            n_households_in_grid_cell = self.n_household_in_grid_cell,
            admin_name=self.geom_id,
            n=self.n,
            nr_cells_to_assess=self.model.settings['decisions']['migration']['nr_cells_to_assess'],
            smod_mask=self.mask_household_allocation,
            damages_coastal_cells=self.damages_coastal_cells,
            coastal_amenity_cells=coastal_amenity_cells,
            amenity_weight=self.model.settings['decisions']['migration']['amenity_weight'],
            people_indices_per_household=self._people_indices_per_household,
            household_id_per_person=self._household_id_per_person,
            empty_index_stack=self._empty_index_stack,
            empty_index_stack_counter=self._empty_index_stack_counter,
            index_first_persons_in_household=index_first_persons_in_household,
            new_household_sizes=new_household_sizes,
            new_income_percentiles=new_income_percentiles,
            new_household_types=new_household_types,
            new_risk_perceptions=new_risk_perceptions,
            new_risk_aversions=new_risk_aversions,
            gender_movers=people['gender'],
            age_movers=people['age'],
            locations=self._locations,
            size=self._size,
            gender=self._person_attribute_array[0, :],
            age=self._person_attribute_array[1, :],
            
            cells_to_move_to=cells_to_move_to,
            income_percentile=self._income_percentile,
            household_type=self._hh_type,
            risk_perception=self._risk_perception,
            risk_aversion=self._risk_aversion,
            would_have_moved=self._would_have_moved,
            admin_indices=self.geom['properties']['region']['indices'],
            gt=self.geom['properties']['region']['gt'],
        )
        # Not the optimal placement. Maybe include this function in numba_add()
        self.initiate_household_attributes_movers(
            n_movers=new_income_percentiles.size)

        assert (self.size != -1).any()
        self.population += people['from'].size
        # update household density in cells
        self.create_mask_household_allocation()

    @staticmethod
    @njit
    def move_numba(population,
                   n: int,
                   people_indices_per_household: np.ndarray,
                   empty_index_stack: np.ndarray,
                   empty_index_stack_counter: int,
                   indice_cell_agent: np.ndarray,
                   households_to_move: np.ndarray,
                   n_movers: int,
                   move_to_region: np.ndarray,
                   admin_idx: int,
                   locations: np.ndarray,
                   size: np.ndarray,
                   hh_type: np.ndarray,
                   ead: np.ndarray,
                   ead_dryproof: np.ndarray,
                   gender: np.ndarray,
                   age: np.ndarray,
                   risk_aversion: np.ndarray,
                   income_percentile: np.ndarray,
                   income: np.ndarray,
                   wealth: np.ndarray,
                   risk_perception: np.ndarray,
                   flood_timer: np.ndarray,
                   adapt: np.ndarray,
                   time_adapt: np.ndarray,
                   adaptation_costs: np.ndarray,
                   decision_horizon: np.ndarray,
                   property_value: np.ndarray,
                   amenity_value: np.ndarray,
                   beach_proximity_bool: np.ndarray,
                   beach_amenity: np.ndarray,
                   distance_to_coast: np.ndarray,
                   would_have_moved: np.ndarray,
                   ) -> tuple[int,
                              int,
                              np.ndarray,
                              np.ndarray,
                              np.ndarray,
                              np.ndarray,
                              np.ndarray,
                              np.ndarray]:
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
        assert np.all(
            households_to_move[:-1] >= households_to_move[1:])  # ensure array is in descending order
        max_stack_counter = empty_index_stack.size
        from_region = np.full(n_movers, admin_idx, dtype=np.int16)
        to_region = np.full(n_movers, -1, dtype=np.int16)
        gender_movers = np.full(n_movers, -1, dtype=np.int8)
        age_movers = np.full(n_movers, -1, dtype=np.int8)
        household_id = np.full(n_movers, -1, dtype=np.int32)

        # Household level attributes
        risk_aversion_movers =  np.full(
            households_to_move.size, -1, dtype=np.float32)
        income_percentile_movers = np.full(
            households_to_move.size, -1, dtype=np.int16)
        risk_perception_movers = np.full(
            households_to_move.size, -1, dtype=np.float32)
        household_income_movers = np.full(
            households_to_move.size, -1, dtype=np.float32)
        household_type_movers = np.full(
            households_to_move.size, -1, dtype=np.int8)

        k = 0
        for i in range(households_to_move.size):
            household_to_move = households_to_move[i]
            households_size = size[household_to_move]
            move_to = move_to_region[i]

            # Household level attributes
            risk_aversion_movers[i] = risk_aversion[household_to_move]
            income_percentile_movers[i] = income_percentile[household_to_move]
            risk_perception_movers[i] = risk_perception[household_to_move]
            household_income_movers[i] = income[household_to_move]
            household_type_movers[i] = hh_type[household_to_move]

            for j in range(households_size):
                to_region[k] = move_to
                household_id[k] = i
                # index of the current mover
                person_index = people_indices_per_household[household_to_move, j]
                assert person_index != -1
                # set gender in move_dictionary
                gender_movers[k] = gender[person_index]
                # set risk aversion in move_dictionary
                age_movers[k] = age[person_index]

                # reset values for person
                gender[person_index] = -1
                age[person_index] = -1

                assert empty_index_stack[empty_index_stack_counter] != -1
                # increment self._empty_index_stack_counter
                empty_index_stack_counter += 1
                assert empty_index_stack_counter < max_stack_counter
                # just check if index stack is indeed empty
                assert empty_index_stack[empty_index_stack_counter] == -1
                empty_index_stack[empty_index_stack_counter] = person_index

                k += 1
            # Shifting household positions so the last positions in the array
            # are removed           
            size[household_to_move] = size[n - 1]
            size[n - 1] = -1
            hh_type[household_to_move] = hh_type[n - 1]
            hh_type[n - 1] = -1          
            locations[household_to_move] = locations[n - 1]
            locations[n - 1] = np.nan
            ead[household_to_move] = ead[n - 1]
            ead[n - 1] = -1
            ead_dryproof[household_to_move] = ead_dryproof[n - 1]
            ead_dryproof[n - 1] = -1
            income_percentile[household_to_move] = income_percentile[n - 1]
            income_percentile[n - 1] = -1
            income[household_to_move] = income[n - 1]
            income[n - 1] = -1
            wealth[household_to_move] = wealth[n - 1]
            wealth[n - 1] = -1
            risk_perception[household_to_move] = risk_perception[n - 1]
            risk_perception[n - 1] = -1
            flood_timer[household_to_move] = flood_timer[n - 1]
            flood_timer[n - 1] = -1
            adapt[household_to_move] = adapt[n - 1]
            adapt[n - 1] = 0
            time_adapt[household_to_move] = time_adapt[n - 1]
            time_adapt[n - 1] = -1
            adaptation_costs[household_to_move] = adaptation_costs[n - 1]
            adaptation_costs[n - 1] = -1
            decision_horizon[household_to_move] = decision_horizon[n - 1]
            decision_horizon[n - 1] = -1
            risk_aversion[household_to_move] = risk_aversion[n - 1]
            risk_aversion[n - 1] = -1
            property_value[household_to_move] = property_value[n - 1]
            property_value[n - 1] = -1
            amenity_value[household_to_move] = amenity_value[n - 1]
            amenity_value[n - 1] = -1
            distance_to_coast[household_to_move] = distance_to_coast[n - 1]
            distance_to_coast[n-1] = -1
            indice_cell_agent[household_to_move] = indice_cell_agent[n - 1]
            indice_cell_agent[n-1] = -1
            would_have_moved[household_to_move] = would_have_moved[n - 1]
            would_have_moved[n-1] = -1
            beach_proximity_bool[household_to_move] = beach_proximity_bool[n - 1]
            beach_proximity_bool[n - 1] = -1

            beach_amenity[household_to_move] = beach_amenity[n - 1]
            beach_amenity[n - 1] = -1

            if household_to_move != n - 1:
                # now we move the last household to the empty spot in the
                # people indices
                people_indices_per_household[household_to_move] = people_indices_per_household[n - 1]

            # and finally reset the last spot of the
            # people_indices_per_household
            people_indices_per_household[n - 1] = -1

            n -= 1
        population -= n_movers

        return population, n, empty_index_stack_counter, from_region, to_region, household_id, gender_movers, age_movers, risk_aversion_movers, income_percentile_movers, household_type_movers, household_income_movers, risk_perception_movers

    def create_dict_decision_params(self):

        # Fix risk perception at zero for a scenario of no flood perception
        if not self.model.settings['general']['dynamic_behavior'] and not self.model.spin_up_flag:
            self.risk_perception *= 0

        # Collect all params in dictionary
        decision_params = {
            'geom_id': self.geom_id,
            'loan_duration': self.model.settings['adaptation']['loan_duration'],
            'expendature_cap': self.model.settings['adaptation']['expenditure_cap'],
            'lifespan_dryproof': self.model.settings['adaptation']['lifespan_dryproof'],
            'n_agents': self.n,
            'sigma': self.model.settings['decisions']['risk_aversion'],
            'wealth': self.wealth,
            'income': self.income,
            'amenity_value': self.amenity_value,
            'amenity_weight': self.model.settings['decisions']['migration']['amenity_weight'],
            'p_floods': 1 / np.array(
                [
                    key for key in self.model.data.inundation_maps_hist.keys()]),
            'risk_perception': self.risk_perception,
            'expected_damages': self.damages,
            'expected_damages_adapt': self.damages_dryproof_1m,
            'adaptation_costs': self.adaptation_costs,
            'premium': self.insurance_premium,
            'adapted': self.adapt,
            'time_adapted': self.time_adapt,
            'T': self.decision_horizon,
            'r': self.model.settings['decisions']['time_discounting']}
    
        return decision_params

    def calculate_utility_of_no_actions(self, decision_params):
        # Calculate EU of adaptation or doing nothing
        EU_do_nothing = self.model.agents.decision_module.calcEU_do_nothing(**decision_params)
        assert (EU_do_nothing != -1).all()
        return EU_do_nothing

    def calculate_utility_of_dry_flood_proofing(self, decision_params):       
        # Calculate EU of adaptation (set to -inf if we want to exclude
        # this behavior)

        if 'government_strategy' in self.model.args:
            strategy = self.model.args.government_strategy
        else:
            strategy = self.model.settings['adaptation']['government_strategy']
        include_adaptation = self.model.settings['agent_behavior']['include_adaptation'] and strategy != 'no_adaptation'

        if include_adaptation or self.model.spin_up_flag:
            EU_adapt = self.model.agents.decision_module.calcEU_adapt(**decision_params)
            EU_adapt_copy = EU_adapt.copy()
            assert (EU_adapt != -1).all()
        else:
            EU_adapt = self.model.agents.decision_module.calcEU_adapt(**decision_params)
            EU_adapt_copy = EU_adapt.copy()
            # Household can no longer implement adaptation after the
            # spin-up period
            EU_adapt[np.where(self.adapt != 1)] = -np.inf
        return EU_adapt, EU_adapt_copy

    def calculate_utility_of_insuring(self, decision_params):
        if self.model.settings['agent_behavior']['include_insurance']:
            EU_insure = self.model.agents.decision_module.calcEU_insure(**decision_params)
            assert (EU_insure != -1).all()
        else:
            EU_insure = np.full(self.n, -np.inf)
        return EU_insure

    def calculate_utility_of_migration(self, regions_select):
        if (self.model.settings['agent_behavior']['include_migration']):
            # Determine EU of migration and which region yields the highest
            # EU

            income_distribution_regions = np.array(
                self.agents.regions.income_distribution_region, dtype=np.int32)

            # get damage factor coastal cells 

            EU_migr_MAX, ID_migr_MAX, cells_assessed = self.agents.decision_module.EU_migrate(
                property_value_nodes = self.agents.regions.property_value_nodes,
                current_amenity_premium = self.coastal_amenity_premium,
                average_ead_snapshot = np.array(self.agents.regions.average_ead_snapshot),
                snapshot_damages_cells = self.agents.regions.snapshot_damages_cells,
                risk_perception = self.risk_perception,
                admin_idx = self.admin_idx,
                geom_id = self.geom_id,
                regions_select=regions_select,
                n_agents=self.n,
                sigma=self.model.settings['decisions']['risk_aversion'],
                wealth=self.wealth,
                income_distribution_regions=income_distribution_regions,
                income_percentile=self.income_percentile,
                amenity_premium_regions=self.model.agents.regions.snapshot_amenity_premium_cells,
                amenity_weight=self.model.settings['decisions']['migration']['amenity_weight'],
                distance=self.distance_vector,
                T=self.decision_horizon,
                r=self.model.settings['decisions']['time_discounting'],
                Cmax=self.fixed_migration_cost * 2,
                cost_shape=self.model.settings['decisions']['migration']['cost_shape']
            )

            # assert (EU_migr_MAX != -1).all()

        else:
            n_choices = np.min([10, regions_select.size])
            EU_migr_MAX = np.full((n_choices, self.n), -np.inf, np.float32)
            ID_migr_MAX = np.full((n_choices, self.n), self.admin_idx, np.int32)
            cells_assessed =  np.full(
                (self.model.agents.regions.n, self.n), -1, dtype=np.int32)

        return EU_migr_MAX, ID_migr_MAX, cells_assessed

    def decide_household_strategy(
            self,
            EU_do_nothing,
            EU_adapt,
            EU_insure,
            EU_migr_MAX
            ):

        # later replace these repeated comparisons with argmax()?
        #### Compare migration decisions ####
        households_intending_to_migrate =  np.where(
            np.logical_and(
                EU_migr_MAX > EU_adapt,
                EU_migr_MAX > EU_do_nothing,
                EU_migr_MAX > EU_insure
                )
            )
        
        # randomly convert migration intention to migration behavior
        households_to_move = self.model.random_module.random_state.choice(
            households_intending_to_migrate[0],
            int((self.model.settings['decisions']['migration']['intention_to_behavior']) *
                households_intending_to_migrate[0].size),
            replace=False)
        households_to_move = np.sort(households_to_move)[::-1]

        # get households not moving
        households_intention_lost = np.setdiff1d(households_intending_to_migrate, households_to_move)

        # set eu of migration for households that did not migrate to -np.inf
        mask = np.full(EU_migr_MAX.size, True)
        mask[households_to_move] = False
        EU_migr_MAX[mask] = -np.inf

        # again compare EU flood proofing decisions 
        households_implementing_dry_floodproofing = np.where(
            np.logical_and(
                EU_adapt > EU_do_nothing,
                EU_adapt >= EU_insure,
                EU_adapt >= EU_migr_MAX
                )
            )
        # compare EU insurance decisions 
        households_taking_up_insurance =  np.where(
            np.logical_and(
                EU_insure > EU_do_nothing,
                EU_insure > EU_adapt,
                EU_insure > EU_migr_MAX
                )
            )

        return households_implementing_dry_floodproofing, households_taking_up_insurance, households_to_move, households_intention_lost

    def process_household_decisions(
            self,
            households_implementing_dry_floodproofing,
            households_taking_up_insurance,
            ):

        #### first process dry floodproofing decisions ####
        # set adaptation status to 1
        self.adapt[households_implementing_dry_floodproofing] = 1
        # Check which people will adapt and whether they made this
        # decision for the first time
        pos_first_time_adapt = (self.adapt == 1) * (self.time_adapt == 0)

        # some test
        if not self.model.spin_up_flag and self.model.args.government_strategy == 'no_adaptation':
            assert np.sum(pos_first_time_adapt) == 0

        # store money spend on adaptation measures
        self.household_spendings = pos_first_time_adapt.sum() * self.adaptation_cost
        gdp = self.agents.GDP_change.GDP_country[self.geom_id[:3]].loc[self.model.current_time.year]
        self.household_spendings_relative_to_gdp = self.household_spendings / gdp

        # Set the timer for these people to 0 when outside spinup. Set to random number within range when inside spin up
        if self.model.spin_up_flag:
            self.time_adapt[pos_first_time_adapt] = self.model.random_module.random_state.integers(1, self.model.settings['adaptation']['lifespan_dryproof'], np.sum(pos_first_time_adapt))
        else: 
            self.time_adapt[pos_first_time_adapt] = 1

        # Update timer for next year
        self.time_adapt[self.time_adapt != 0] += 1 # this also update new households 

        # Update the percentage of households implementing flood proofing
        # Check for missing data
        self.n_households_adapted = np.sum(self.adapt[self.adapt == 1])
        # if self.n_households_exposed > 0:
        #     self.percentage_adapted = round(
        #         self.n_households_adapted / self.n_households_exposed * 100, 2)
        # else:
        #     self.percentage_adapted = 0

        ### process insurance decisions ####
        # first reset insurance status to 0
        self.adapt[self.adapt == 2] = 0

        # set adaptation status to 2
        self.adapt[households_taking_up_insurance] = 2
        self.percentage_insured = round(
            np.sum(self.adapt == 2) / self.adapt.size * 100, 2)
        self.n_households_insured = np.sum(self.adapt[self.adapt == 2])
        self.households_taking_up_insurance = households_taking_up_insurance

    def iterate_over_regions(
            self,
            frac_total_pop_in_node,
            households_to_move,
            move_to_region,
            agents_in_regions,
            ):
        '''This function is used to make sure the number of movers does not exceed a set threshold of n household growth for the recieving regions'''
        # get max growth from settings
        if self.model.settings['decisions']['migration']['limit_admin_growth']:
            max_growth = self.model.settings['decisions']['migration']['max_admin_growth']
        else:
            max_growth = 1E6 # just a high number to make sure all households can move
        
        # create empty arrays to fill
        households_not_to_move_subset = np.array([], np.int32)

        # iterate over regions in destination
        regions, counts = np.unique(move_to_region, return_counts=True)
        for region, count in zip(regions, counts):
            # get n available housing
            if self.model.settings['decisions']['migration']['limit_admin_growth']:
                n_available_housing = self.agents.regions.n_available_housing[region]
            else: n_available_housing = np.inf
            # get n regions in country
            n_regions_in_country = len(self.agents.population_change.admins_iso3[self.geom_id[:3]])
            
            # get max growth
            frac_total_pop_in_node_adusted = max(frac_total_pop_in_node, 1E-3) # set to at least 0.1 percent to account for small floodplains

            if self.agents.regions.ids[region].endswith('flood_plain'):
                n_max_growth = int(min([agents_in_regions[region] * max_growth *  frac_total_pop_in_node_adusted, n_available_housing * frac_total_pop_in_node]))
            else:
                n_max_growth = int(min([agents_in_regions[region] * max_growth *  frac_total_pop_in_node_adusted, n_available_housing * frac_total_pop_in_node]))
                # n_max_growth = np.inf
            if count > n_max_growth:
                # print(f'max growth exceeded by {count - n_max_growth} for migration from {self.geom_id}')
                # randomly select agents which are allowed to migrate region region 
                households_moving_to_region = households_to_move[np.where(move_to_region == region)]
                # get the maximum number of agents allowed to move to region
                n_households_not_to_move = households_moving_to_region.size - n_max_growth
                n_households_not_to_move = np.min([n_households_not_to_move, households_moving_to_region.size]) # make sure not to select more agents than moving
                # sample households that are allowed to move
                households_not_to_move_subset = np.concatenate([households_not_to_move_subset, self.model.random_module.random_state.choice(households_moving_to_region, n_households_not_to_move, replace=False)])

        return households_not_to_move_subset
    
    def iterate_over_second_best_choices(
            self,
            frac_total_pop_in_node,
            agents_in_regions,
            choice,
            ID_migrate,
            EU_migrate,
            EU_do_nothing,
            EU_adapt,
            EU_insure,
            households_already_moved
        ):
        # set EU of migration to -np.inf for households that could migrate to previous best choice
        EU_migrate[choice, households_already_moved] = -np.inf

        # make decision again for households but then consider second best option
        # assess strategies again, taking into account second best destination
        households_implementing_dry_floodproofing_choice, households_taking_up_insurance_choice, households_to_move_to_choice, households_intention_lost_choice = self.decide_household_strategy(
                EU_do_nothing,
                EU_adapt,
                EU_insure,
                EU_migrate[choice, :]) # now select second choice here
    
        move_to_region_choice = ID_migrate[choice, households_to_move_to_choice]
        EU_migrate[:, households_intention_lost_choice] = -np.inf

        # again iterate over regions to check if full
        households_not_to_move_to_choice = self.iterate_over_regions(
            frac_total_pop_in_node = frac_total_pop_in_node,
            households_to_move=households_to_move_to_choice,
            move_to_region=move_to_region_choice,
            agents_in_regions=agents_in_regions
            )
        # combine
        households_to_move_choice = np.setdiff1d(households_to_move_to_choice, households_not_to_move_to_choice) 
        assert np.intersect1d(households_already_moved, households_to_move_to_choice).size == 0
   
        return households_implementing_dry_floodproofing_choice, households_taking_up_insurance_choice, households_to_move_choice, households_not_to_move_to_choice

    def distribute_movers(
            self,
            EU_do_nothing,
            EU_adapt,
            EU_insure,
            EU_migrate, 
            size,
            ID_migrate,
            agents_in_regions,
        ):
        
        '''This function is used to make sure the number of movers does not exceed a set threshold of population growth for the recieving regions'''
        # get fraction of total population in node to determine share of migration to node
        frac_total_pop_in_node = self.population/ self.population_in_country

        # first compare strategies
        households_implementing_dry_floodproofing, households_taking_up_insurance, households_to_move, households_intention_lost_first_choice = self.decide_household_strategy(
            EU_do_nothing,
            EU_adapt,
            EU_insure,
            EU_migrate[0, :]) # only select first choice here
    
        # get initial move to region
        move_to_region = ID_migrate[0, households_to_move]

        # store initinal n movers
        n_movers_init = households_to_move.size

        # set migration EU for households losing intention to migrate to -np.inf for all regions
        EU_migrate[:, households_intention_lost_first_choice] = -np.inf

        # iterate over regions to determine if regions are full and if so remove households from move array
        households_not_to_move_choice = self.iterate_over_regions(
            frac_total_pop_in_node = frac_total_pop_in_node,
            households_to_move=households_to_move,
            move_to_region=move_to_region,
            agents_in_regions=agents_in_regions
        )
        households_to_move = np.setdiff1d(households_to_move, households_not_to_move_choice)  

        # get destination
        move_to_region = ID_migrate[0, households_to_move]
        
        # init choice
        max_choice = EU_migrate.shape[0] - 1
        choice = 0
        move_to_choice = np.zeros(self.n, np.int32)
        # iterate over regions to allocate as many households as posible
        while households_not_to_move_choice.size > 0 and choice < max_choice:
            choice += 1
            households_implementing_dry_floodproofing_choice, households_taking_up_insurance_choice, households_to_move_to_choice, households_not_to_move_choice  = self.iterate_over_second_best_choices(
                frac_total_pop_in_node,
                agents_in_regions,
                choice,
                ID_migrate,
                EU_migrate,
                EU_do_nothing,
                EU_adapt,
                EU_insure,
                households_to_move
                )  
            
            households_to_move = np.concatenate([households_to_move, households_to_move_to_choice])
            move_to_choice[households_to_move_to_choice] = choice
            # get which households are allowed to move into their second best destination and fill move to region array
            move_to_region = np.concatenate([move_to_region, ID_migrate[choice, households_to_move_to_choice]])
            
            households_implementing_dry_floodproofing = np.concatenate([
                households_implementing_dry_floodproofing,
                households_implementing_dry_floodproofing_choice])

            households_taking_up_insurance = np.concatenate([
                households_taking_up_insurance,
                households_taking_up_insurance_choice])
            
        if households_not_to_move_choice.size > 0:
            print(f'{households_not_to_move_choice.size} households could not be allocated moving from {self.geom_id}')
        # again compare utilities to allow households not moving to adapt
        households_implementing_dry_floodproofing_final, households_taking_up_insurance_subset_final, _, _ = self.decide_household_strategy(
            EU_do_nothing,
            EU_adapt,
            EU_insure,
            np.full(self.n, -np.inf, np.float32)) # now set migration to -np.inf to only compare stay + stay adapt. Households that adapt but were moving will be removed from arrays anyway. 

        # combine the arrays of households implementing dry floodproofing and taking up insurance
        households_implementing_dry_floodproofing = np.unique(np.concatenate([
            households_implementing_dry_floodproofing,
            households_implementing_dry_floodproofing_final]))
        
        households_taking_up_insurance = np.unique(np.concatenate([
            households_taking_up_insurance,
            households_taking_up_insurance_subset_final]))
       
        # sort and process
        sort_idx = np.argsort(households_to_move)[::-1]
        households_to_move = households_to_move[sort_idx]
        move_to_region = move_to_region[sort_idx]
        households_sizes = size[households_to_move]
        n_movers = households_sizes.sum()
        
        # print(f'{households_to_move.size} of {n_movers_init} moved from {self.geom_id}') # DEBUGGING
        self.fraction_intending_to_migrate = households_to_move.size / self.n
        self.n_intending_to_migrate = households_to_move.size

        return move_to_choice, households_implementing_dry_floodproofing, households_taking_up_insurance,households_to_move, move_to_region, households_sizes, n_movers

    def move(self):
        '''This function processes the individual household agent decisions. It calls the functions to calculate
        expected utility of stayin, adapting, and migrating to different regions. Agents that decide to move are
        then removed from the arrays and stored in move dictionaries'''
        # Reset counters
        self.n_moved_out_last_timestep = 0
        self.n_moved_in_last_timestep = 0
        self.people_moved_out_last_timestep = 0
        self.perc_people_moved_out = 0
        self.fraction_intending_to_migrate = 0
        self.n_intending_to_migrate = 0

        # Run some checks to assert all households have attribute values
        # assert (sigma != -1).all()
        if (self.income <= -1).all() or (self.wealth <= -1).all() or (self.risk_perception <= -1).all() or (self.decision_horizon <= -1).all():
            self.model.logger.info(f'Some households have missing attribute values in {self.geom_id}')
            raise ValueError(f'Some households have missing attribute values in {self.geom_id}')
        
        # Not used in decisions, all households currently have the same risk
        # aversion setting (sigma).
        assert (self.risk_aversion != -1).all()

        # Reset timer and adaptation status when lifespan of dry proofing is
        # exceeded
        self.adapt[self.time_adapt ==
                   self.model.settings['adaptation']['lifespan_dryproof']] = 0
        # People have to make adaptation choice again.
        self.time_adapt[self.time_adapt ==
                        self.model.settings['adaptation']['lifespan_dryproof']] = 0

        # Only select region for calculations if agents present
        if self.n>0:
            
            # collect params
            decision_params = self.create_dict_decision_params()

            # get regions to move to
            regions_select = self.select_regions_where_to_move()

            # sample error terms used to account for location and time specific unobserved preferences in staying and migrating
            self.agents.decision_module.sample_error_terms(self.n, regions_select)
            
            # calculate utilities
            EU_do_nothing = self.calculate_utility_of_no_actions(decision_params)
            EU_adapt, EU_adapt_copy = self.calculate_utility_of_dry_flood_proofing(decision_params)
            EU_insure = self.calculate_utility_of_insuring(decision_params)
            EU_migrate, ID_migrate, cells_assessed = self.calculate_utility_of_migration(regions_select)

            # simply execute strategy - for DEBUGGING
            # ######################################################
            # households_implementing_dry_floodproofing, households_taking_up_insurance, households_to_move, households_intention_lost = self.decide_household_strategy(
            #     EU_do_nothing,
            #     EU_adapt,
            #     EU_insure,
            #     EU_migrate[0, :]) # only select first choice here
            
            # households_sizes = self.size[households_to_move]
            # n_movers = households_sizes.sum()
            # move_to_region = ID_migrate[0, households_to_move]
            # assert self.admin_idx not in move_to_region
            ######################################################

            # execute strategy accounting for limit population growth as defined in settings.yml
            move_to_choice, households_implementing_dry_floodproofing, households_taking_up_insurance,households_to_move, move_to_region, households_sizes, n_movers = self.distribute_movers(
                EU_do_nothing = EU_do_nothing,
                EU_adapt = EU_adapt,
                EU_insure = EU_insure,
                EU_migrate=EU_migrate.copy(),
                size=self.size,
                ID_migrate=ID_migrate,
                agents_in_regions=self.agents.regions.agents_in_simulation)

            cells_to_move_to = cells_assessed[move_to_region, households_to_move].astype(np.int64)
            assert all(ID_migrate[move_to_choice[households_to_move], households_to_move] == move_to_region)
            EU_migrate_movers = EU_migrate[(move_to_choice[households_to_move], households_to_move)]
            households_that_would_have_adapted = np.where(EU_migrate_movers < EU_adapt_copy[households_to_move])
            households_that_have_not_yet_adapted = np.where(self.time_adapt == 0)[0]
            households_that_would_have_moved = np.where(np.logical_and(
                EU_migrate[move_to_choice, np.arange(self.n)][households_that_have_not_yet_adapted] < EU_adapt[households_that_have_not_yet_adapted],
                EU_migrate[move_to_choice, np.arange(self.n)][households_that_have_not_yet_adapted] > EU_do_nothing[households_that_have_not_yet_adapted]),
                )[0]

            self.n_households_that_would_have_adapted = households_that_would_have_adapted[0].size
            self.n_people_that_would_have_adapted = self.size[households_that_would_have_adapted].sum()

            # record would have moved
            self.would_have_moved[households_that_would_have_moved] = 1

            self.n_households_that_would_have_moved = households_that_would_have_moved.size
            self.n_people_that_would_have_moved = self.size[households_that_would_have_moved].sum()

            # if no households are moving, return None
            if households_to_move.size == 0:
                if self.model.args.low_memory_mode:
                    delattr(self, 'damages')
                    delattr(self, 'damages_dryproof_1m')
                return None, None

            # process the adaptation decisions
            self.process_household_decisions(
                households_implementing_dry_floodproofing,
                households_taking_up_insurance,
                )
        else:
            # If 0 households are present return None
            if self.model.args.low_memory_mode:
                delattr(self, 'damages')
                delattr(self, 'damages_dryproof_1m')
            return None, None

        # set some tracking variables
        self.n_moved_out_last_timestep = households_sizes.size
        self.people_moved_out_last_timestep = n_movers
        self.perc_people_moved_out = (n_movers / self.population) * 100
        if households_to_move.size > 0:
            # store fraction of adapted in movers
            self.n_adapted_movers = np.sum(self.adapt[households_to_move])
            self.fraction_adapted_movers = np.mean(self.adapt[households_to_move])
            # subtract adapted movers from avoided migration (cumulative this would still add up)
            # self.n_people_that_would_have_moved -= self.size[households_to_move][self.adapt[households_to_move] == 1].sum()
            self.n_households_that_would_have_moved -= self.would_have_moved[households_to_move].sum()
            self.n_people_that_would_have_moved -= self.size[households_to_move][self.would_have_moved[households_to_move] == 1].sum()

        else:
            self.n_adapted_movers = 0
            self.fraction_adapted_movers = 0
        # Assert nobody is moving to their own region
        assert not any(move_to_region == self.admin_idx)

        # remove damage arrays (no longer needed, will be updated in next timestep)
        if self.model.args.low_memory_mode:
            delattr(self, 'damages')
            delattr(self, 'damages_dryproof_1m')

        # DEBUGGING
        if self.model.args.run_with_checks:
            # calculate percentage moving to coastal vs inland node
            n_flooded = self.flooded[households_to_move].sum()
            n_movers_coastal = np.sum([self.agents.regions.ids[i].endswith('flood_plain') for i in move_to_region])
            if n_movers_coastal > 0:
                perc_move_to_coast = round(n_movers_coastal/ move_to_region.size*100)
                if n_flooded > 0:
                    print(f'percentage movers from {self.geom_id} moving to a floodplain: {perc_move_to_coast}% [{n_movers_coastal} of {move_to_region.size}] *flooded')
                else:
                    print(f'percentage movers from {self.geom_id} moving to a floodplain: {perc_move_to_coast}% [{n_movers_coastal} of {move_to_region.size}]')
            else:
                perc_move_to_coast = 0
                if n_flooded > 0:
                    print(f'percentage movers from {self.geom_id} moving to a floodplain: {perc_move_to_coast}% [{n_movers_coastal} of {move_to_region.size}] *flooded')
                else:
                    print(f'percentage movers from {self.geom_id} moving to a floodplain: {perc_move_to_coast}% [{n_movers_coastal} of {move_to_region.size}]')

            ## check if selected cell in destination is correct
            # get households moving to coastal node
            moving_to_coastal_mask = [self.agents.regions.ids[i].endswith('flood_plain') for i in move_to_region]
            households_to_move_coastal = households_to_move[moving_to_coastal_mask]
            move_to_region_coastal = move_to_region[moving_to_coastal_mask]
            cells_to_move_to_coastal = cells_to_move_to[moving_to_coastal_mask]
            if households_to_move_coastal.size > 0:
            # get random agent
                idx = np.random.random_integers(0, households_to_move_coastal.size-1) 
                household_id = households_to_move_coastal[idx]
                current_ead = self.ead_agents[household_id]
                current_amenity_premium = self.coastal_amenity_premium[household_id]
                # # get future ead
                # # move to region
                move_to = move_to_region_coastal[idx]
                cell_to_move_to = cells_to_move_to_coastal[idx]
                future_ead = self.agents.regions.all_households[move_to].damages_coastal_cells[cell_to_move_to]
                future_amenity_premium = self.agents.regions.all_households[move_to].coastal_amenity_premium_cells[cell_to_move_to]
                assert future_amenity_premium <= current_amenity_premium
                        
        
        # apply move numba function to remove movers from coastal node
        self.population, self.n, self._empty_index_stack_counter, from_region, to_region, household_id, gender, age, risk_aversion, income_percentile, household_type, income, risk_perception = self.move_numba(
            population=self.population,
            n=self.n,
            people_indices_per_household=self._people_indices_per_household,
            empty_index_stack=self._empty_index_stack,
            empty_index_stack_counter=self._empty_index_stack_counter,
            indice_cell_agent=self.indice_cell_agent,
            households_to_move=households_to_move,
            n_movers=n_movers,
            move_to_region=move_to_region,
            admin_idx=self.admin_idx,
            locations=self._locations,
            size=self._size,
            hh_type=self._hh_type,
            ead=self._ead,
            ead_dryproof=self._ead_dryproof,
            gender=self._person_attribute_array[0, :],
            age=self._person_attribute_array[1, :],
            risk_aversion=self._risk_aversion,
            income_percentile=self._income_percentile,
            income=self._income,
            wealth=self._wealth,
            risk_perception=self._risk_perception,
            flood_timer=self._flood_timer,
            adapt=self._adapt,
            adaptation_costs=self._adaptation_costs,
            time_adapt=self._time_adapt,
            decision_horizon=self._decision_horizon,
            property_value=self._property_value,
            amenity_value=self._amenity_value,
            beach_proximity_bool=self._beach_proximity_bool,
            beach_amenity=self._beach_amenity,
            distance_to_coast=self._distance_to_coast,
            would_have_moved=self._would_have_moved
        )
       
        # Return move dictionary      
        people =  {
            "from": np.full(
                n_movers,
                self.admin_idx,
                dtype=np.int16),
            "to": to_region,
            "household_id": household_id,
            "gender": gender,
            "age": age,
        }

        households = {
            "from": np.full(
                np.unique(household_id).size,
                self.admin_idx,
                dtype=np.int16),
            "to": move_to_region,
            "household_id": np.unique(household_id),
            "risk_aversion": risk_aversion,
            "income_percentile": income_percentile,
            "household_type": household_type,
            "income": income,
            "risk_perception": risk_perception,
            "cells_to_move_to": cells_to_move_to
            }

        # set another tracking variable
        if self.population > 0:
            self.percentage_moved = np.round(n_movers/ self.population*100, 2)
        
        # and update mask
        self.create_mask_household_allocation()

        # if migration is not considered return None, else return move dictionaries
        if self.model.settings['agent_behavior']['include_migration']:
            return people, households
        else:
            return None, None