import numpy as np
import pandas as pd
from scipy import interpolate
import random

class FloodRisk():
    def __init__(self, model):
        self.model = model
        self._initialize_admin_cell_indices()
        self.update_risk_admin_cells()
        np.random.seed(0)
        self.model.set_seed(0)

    def _initialize_admin_cell_indices(self):
        all_locs_admin_cells = [household.locations_admin_cells for household in self.model.agents.regions.all_households if household.geom_id.endswith('flood_plain')]
        all_locs_admin_cells_concated = np.concatenate(all_locs_admin_cells, axis = 0)
        
        # Create dictionary for easy query
        admin_names = [household.geom_id for household in self.model.agents.regions.all_households if household.geom_id.endswith('flood_plain')]
        
        indice_query = {}
        for i, name in enumerate(admin_names):
            if i == 0:
                indice_query[name] = np.arange(all_locs_admin_cells[i].shape[0])
            else:
                n_cells = all_locs_admin_cells[i].shape[0]
                last_index_previous = indice_query[admin_names[i-1]][-1]
                # Account for admins with a single cell in the floodplain
                if n_cells == 1:
                    indice_query[name] = np.array([last_index_previous + 1])
                else:
                    indice_query[name] = np.arange(last_index_previous + 1, last_index_previous + n_cells + 1)
            assert indice_query[name].shape[0] == all_locs_admin_cells[i].shape[0]

        # store
        self.indice_query = indice_query
        self.all_locs_admin_cells_concated = all_locs_admin_cells_concated

     
    def check_model_status(self):
        if not self.model.spin_up_flag:
            np.random.seed(random.randint(0, 1E6))
            self.model.set_seed(random.randint(0, 1E6))


    def update_risk_admin_cells(self):
        '''
        This function reads and samples ead for each cell that lies within the floodzone. It stores the sampled values in a dict for easy access.
        The main reason I do this here instead of in the coastal nodes is so this operation is performend only once per timestep.
        '''
        # Use concatenated array to sample from netcdf
        if self.model.args.rcp == 'control':
            current_ead_cells = self.model.data.ead_map.sample_coords(self.all_locs_admin_cells_concated, dt = self.model.config['general']['start_time'])
        else:
            current_ead_cells = self.model.data.ead_map.sample_coords(self.all_locs_admin_cells_concated, dt = self.model.current_time)

        # use indexing to store results in dict
        ead_admin_cells = {}
        for admin in self.indice_query:
            ead_admin_cells[admin] = current_ead_cells[self.indice_query[admin]]

        # Store
        self.ead_admin_cells = ead_admin_cells
    
    @staticmethod
    def sample_water_level(
        locations: np.ndarray, 
        return_periods: np.ndarray, 
        fps: int, 
        inundation_maps: list, 
        rcp: str, 
        start_year: int, 
        current_year: int
        ) -> dict:
        '''
        This function creates a dictionary of water levels for inundation events of different return periods.
        It uses the sample coordenates method of the ArrayReader class instances loaded in data.py. The inundation maps
        are selected based on the scenario defined in the terminal command 'rcp'.
        
        Args:
            locations: an array containing the coordinates of each agent.
            return_periods: an array containing the return periods of flood events included in this model.
            fps: flood protection standard of the admin region
            inundation_maps: list containing the historical and future inundation maps (both as honeybees.ArrayReader objects) under the RCP applied in the run.
            rcp: RCP scenario applied in the model run.
            start_year: starting year of the model run
            current_year: current year in the current timestep
        Returns:
            water_level: a dictionary containing numpy arrays of inundation levels for each agent associated the different return periods.

        '''
        
        water_level_hist = {}
        water_level_2080 = {}

        # Fill water levels by sampling agent locations
        
        for i in return_periods:
            water_level_hist[i] =  inundation_maps[0][i].sample_coords(locations, cache=True)

        if rcp == 'control':
            water_level_2080 = water_level_hist
        else:
            for i in return_periods:
                water_level_2080[i] =  inundation_maps[1][i].sample_coords(locations, cache=True)
        
        # Interpolate water level between year 2000 and 2080
        water_level = {}

        # Extract start time from config
        start_time = start_year
        timestep = current_year - start_time

        # Derive current water depth based on linear interpolation
        for rt in return_periods:
            water_level_hist[rt][water_level_hist[rt] < 0.001] = 0
            water_level_2080[rt][water_level_2080[rt] < 0.001] = 0
            difference = (water_level_2080[rt] - water_level_hist[rt])/ (2080 - start_time)
            water_level[rt] = water_level_hist[rt] + difference * timestep
        
        # Read protection standard and set inundations levels to 0 if protected
        for rt in water_level:
            if rt < fps:
                water_level[rt] = np.zeros(len(water_level[rt]))

        return water_level

    @staticmethod    
    def calculate_ead(
        n_agents: int, 
        water_level: dict, 
        damage_curves: pd.core.frame.DataFrame, 
        property_value: np.ndarray, 
        return_periods: np.ndarray
        ) -> tuple[dict, dict, np.ndarray, np.ndarray]:

        '''
        This function is used to calculate the expected annual damages (ead) under no adaptation and when implementing dry flood proofing for each agent
        
        Args:
            n_agents: number of agents in the current floodplain.
            water_level: a numpy array containing the inundation levels for each agent associated the different return periods.
            damage_curves: pandas dataframe containing the water levels their associated damage factors under no adaptation, and with implementing dry flood proofing.
            property_value: property value of the agents in the current floodplain.
            return_periods: a numpy array containing the return periods of flood events included in the model run
        
        Returns:
            damages: a numpy array containing the expected damages without adaptation for each flood event and each agent
            damages_dryproof_1m: a numpy array containing the expected damages under dry flood proofing for each flood event and each agent
            ead: a numpy array containing the expected annual damages (ead) for each agent under no adaptation
            ead_dryproof: a numpy array containing the expected annual damages (ead) for each agent under dry flood proofing
        '''

            

        # Interpolate damage factor based on damage curves  
        func_dam = interpolate.interp1d(damage_curves.index, damage_curves['baseline'])
        func_dam_dryproof_1m = interpolate.interp1d(damage_curves.index, damage_curves['dryproof_1m'])
        
        # Indicate maximum damage per household
        max_dam = property_value

        # pre-allocate empty array with shape (n_floods, n_agents) for number of damage levels and number of households
        damages = np.zeros((len(return_periods), n_agents), dtype=np.float32)
        
        for i, rt in enumerate(return_periods):
            water_level[rt][water_level[rt] < 0] = 0
            water_level[rt][water_level[rt] > 6] = 6
            # calculate damage per retun period and store in damage dictory
            # place the damage output in the empty array
            damages[i] = func_dam(water_level[rt]) * max_dam
        
        x = 1 / return_periods
        # calculate ead on damage array along the first axis
        ead = np.trapz(damages, x, axis=0)        
        
        # pre-allocate empty array with shape (3, self.n) for number of damage levels and number of households
        damages_dryproof_1m = np.zeros((len(return_periods), n_agents), dtype=np.float32)
        for i, rt in enumerate(return_periods):
            water_level[rt][water_level[rt] < 0] = 0
            water_level[rt][water_level[rt] > 6] = 6
            # calculate damage per retun period and store in damage dictory
            # place the damage output in the empty array
            damages_dryproof_1m[i] = func_dam_dryproof_1m(water_level[rt]) * max_dam
        x = 1 / return_periods

        # calculate ead on damage array along the first axis
        ead_dryproof = np.trapz(damages_dryproof_1m, x, axis=0)  
        
        # # Sum and update expected damages per node
        # agents_that_adapted = np.where(adaptation_status == 1)
        # agents_not_adapted = np.where(adaptation_status == 0)
        return damages, damages_dryproof_1m, ead, ead_dryproof
    
    @staticmethod
    def stochastic_flood(
        water_levels: dict, 
        return_periods: np.ndarray,
        flooded: np.ndarray, 
        flood_count: np.ndarray, 
        risk_perceptions: np.ndarray,
        flood_timer: np.ndarray, 
        risk_perc_min: float, 
        risk_perc_max: float,
        risk_decr: float, 
        settings: dict, 
        current_year: int, 
        spin_up_flag: bool, 
        flood_tracker: int
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        '''
        This function simulates stochastic flooding using a random number generator. In its current 
        implementation only one flood event can occur per year.
        
        Args:
            water_levels: dictionary containing arrays of inundation levels for each agent associated with various return periods.
            return_periods: an array containing the return periods of flood events included in this model.
            flooded: an array containing the flood status of each agent, with 0 indicating not flooded, and 1 indicating flooded.
            flood_count: an array keeping track of the flood experience of each agent.
            risk_perceptions: an array containing the flood risk perceptions of each agent
            method: random if random draw, or a single year.
        Returns:
            flooded: the updated array containing the flood status of each agent
            flood_count: the updated array containing the flood experiance of each agent
            risk_perceptions: the updated array containing the risk perception of each agent.
            flood_timer: store the year in which the flood event occured
            flood_tracker: store the return period of the simulated flood event'''

        # reset flooded to 0 for all households
        flooded *= 0 
        flood_tracker *= 0 
        
        # update flood timer for all households
        flood_timer += 1
        
        if settings['random_flood']:
            # No flooding in spin up
            if not settings['spin_up_flood'] and spin_up_flag:
                pass
            
            else:
            # Simulate flooding based on random draw
                random_draw = np.random.rand()
                for rt in return_periods:
                    if random_draw < (1/rt):
                        flooded[water_levels[rt]>0] = 1
                        flood_count[water_levels[rt]>3] += 1 # Count times people have experienced waterlevels of more than 3 meters (Not used)
                        flood_timer[water_levels[rt]>0] = 0 # Set flood timer to zero for those who experienced flooding
                        if rt >= settings['flood_protection_standard']:
                            flood_tracker = rt
                        break 
        else:
            year = settings['year']
            rt = settings['rt']
            
            if current_year == year:
                flooded[water_levels[rt]>0] = 1
                flood_count[water_levels[rt]>3] += 1 # Count times people have experienced waterlevels of more than 3 meters (Not used)
                flood_timer[water_levels[rt]>0] = 0 # Set flood timer to zero for those who experienced flooding


        risk_perceptions =  risk_perc_max * 1.6 ** (risk_decr * flood_timer) + risk_perc_min
        return flooded, flood_count, risk_perceptions, flood_timer, flood_tracker
