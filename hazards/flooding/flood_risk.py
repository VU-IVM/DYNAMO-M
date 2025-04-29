import numpy as np
import matplotlib.pyplot as plt
import os

class FloodRisk():
    '''This class contains all the functions that are related to the flood risk model.'''

    def __init__(self, model):
        self.model = model

    def interpolate_water_levels(
        low_memory_mode,
        low_memory_mode_folder,
        cache_file,
        rcp,
        region,
        start_year,
        end_year,
        locations,
        return_periods,
        inundation_maps,
        fps,
        cells_on_coastline,
        plot_water_levels=False,
        sample_coords_from_cache=True,
        load_from_cache=True
        
    ):

        '''Derive water level polynomials for given locations and return periods based on inundation maps.

        Args:
            low_memory_mode (bool): If True, the function will store the water levels in a memmap file to reduce memory usage.
            low_memory_mode_folder (str): The folder in which to store the memmap file.
            cache_file (str): The path to the cache file to store the water levels in. If None, the function will not store the water levels in a cache file.
            rcp (str): The RCP scenario to derive water levels for.
            region (str): The region to derive water levels for.
            start_year (int): The start year of the simulation.
            end_year (int): The end year of the simulation.
            locations (numpy.ndarray): Array of shape (n, 2) containing the (longitude, latitude) coordinates of n locations.
            return_periods (numpy.ndarray): Array of integers representing the return periods to compute polynomials for.
            inundation_maps (list): List of three dictionaries, each containing inundation maps for the four time periods
                (historical, 2030s, and 2080s). The inundation maps should be instances of the `inundation_maps`
                class and should have a method called `sample_coords` that returns water level data for a given set of
                coordinates.

        Returns:
            fp: A memmap file containing the water levels for each location, return period, and year.
            water_levels_admin_cells (numpy.ndarray): Array of shape (n_return_periods, n_locations, n_years) containing the water levels for each location, return period, and year.
        '''

        # perallocate array to store water levels for cells
        time_arr = np.arange(2000, 2182) # allow sim to run untill 2081 with decision horizon of 100 years
        water_levels_admin_cells = np.full((return_periods.size, locations.shape[0], time_arr.size), -1, np.float32)        
        if plot_water_levels: fig, ax = plt.subplots()

        if low_memory_mode:
            folder_name = low_memory_mode_folder
        else:
            folder_name = 'cache'
        if rcp == 'control':
            store_sampled_levels_dir = os.path.join(folder_name, f'water_levels_rcp4p5')
        else:
            store_sampled_levels_dir = os.path.join(folder_name, f'water_levels_{rcp}')
        path_inundation_arrays = os.path.join(store_sampled_levels_dir, f'{region}_inundation.npy')
        if os.path.exists(path_inundation_arrays) and load_from_cache:
            print(f'loading water levels in {region} from cache')
            water_levels_admin_cells = np.load(path_inundation_arrays)
            if rcp == 'control':
                # if in control rcp overwrite all years with first year
                water_levels_admin_cells = np.repeat(water_levels_admin_cells[:, :, 0], water_levels_admin_cells.shape[2]).reshape((water_levels_admin_cells.shape))           
        else:
            # Derive current water depth based on simple interpolation
            for i, rt in enumerate(return_periods):
                water_level_hist = inundation_maps['hist'][rt].sample_coords(
                    locations, cache = sample_coords_from_cache)      

                if not rcp == 'control':
                    water_level_2080 = inundation_maps[2080][rt].sample_coords(
                        locations, cache = sample_coords_from_cache)

                elif rcp == 'control':
                    water_level_2080 = water_level_hist

                for water_level in [water_level_hist, water_level_2080]:
                    water_level[np.isnan(water_level)] = 0 
                    water_level[water_level < 0] = 0
                    assert water_level.min() >= 0

                # check if water levels on the coast increase over time
                if not (water_level_2080[cells_on_coastline] >= water_level_hist[cells_on_coastline]).all():
                    print('water levels 2080 < water levels 2030')
                    water_level_2080[cells_on_coastline] = np.maximum(water_level_2080[cells_on_coastline], water_level_hist[cells_on_coastline])
                    assert (water_level_2080[cells_on_coastline] >= water_level_hist[cells_on_coastline]).all()

                x = np.array([2000, 2080])
                y = np.array([water_level_hist, water_level_2080])

                # simply interpolate linearly between datapoints
                # annual increase hist-2030
                water_level_admin_cells = np.repeat(water_level_hist, time_arr.size).reshape((water_level_hist.size, time_arr.size))

                annual_increase_2000_2080 = (water_level_2080 - water_level_hist)/ 80
                index = 1

                for j, year in enumerate(np.arange(2000, time_arr.max())):
                    water_level_admin_cells[:, index] = water_level_admin_cells[:, index-1] + annual_increase_2000_2080
                    index += 1

                # filter and store
                water_levels_admin_cells[i]  = np.round(np.maximum(water_level_admin_cells, np.repeat(water_level_hist, water_level_admin_cells.shape[1]).reshape(water_level_admin_cells.shape)), 2)
                ### plotting
                if plot_water_levels:
                    ax.plot(time_arr, water_levels_admin_cells[i].max(axis = 0), label = rt)
                    # add datapoints
                    ax.scatter(x, [levels.max() for levels in y], marker = 'x')

            # check if water levels increase over time per return period
            for i in range(len(water_levels_admin_cells) - 1):
                if not (water_levels_admin_cells[i][cells_on_coastline] >= water_levels_admin_cells[i + 1][cells_on_coastline]).all():
                    # get n cells that need to be adjusted
                    n_cells_to_adjust = np.where(water_levels_admin_cells[i][cells_on_coastline] < water_levels_admin_cells[i + 1][cells_on_coastline])[0].size
                    percent_cells_to_adjust = round(n_cells_to_adjust / water_levels_admin_cells[i].size * 100, 2)
                    water_levels_admin_cells[i + 1][cells_on_coastline] = np.minimum(water_levels_admin_cells[i][cells_on_coastline], water_levels_admin_cells[i + 1][cells_on_coastline])
                    print(f'{n_cells_to_adjust} cells ({percent_cells_to_adjust}%) of cells adjusted in {region} of rp {return_periods[i]} under {rcp}')

            # filter out years 2000 - start sim
            water_levels_admin_cells = water_levels_admin_cells[:, :, (start_year - 2000):]
        
            if not os.path.exists(path_inundation_arrays):
                os.makedirs(store_sampled_levels_dir, exist_ok=True)
                np.save(path_inundation_arrays, water_levels_admin_cells)
                       
            if plot_water_levels:
                i_fps = int(np.where(return_periods == fps)[0])
                ax.hlines(y = water_levels_admin_cells[i_fps].max(axis = 0)[15], xmin=2015, xmax=end_year, linestyles='--', colors='k', label='FPS')
                fig.legend()
                ax.set_title(f'Max inundation in {region} {rcp}')
                fig.savefig(f'figures_FPS/{region}_{rcp}.png')
                fig.clear()
                ax.clear()
       
        # create memmap (values for inundation levels are estimated for entire run thus array do not have to be overwritten)
        if cache_file != None:
            folder_name = os.path.dirname(path_inundation_arrays)
            filename = os.path.join(folder_name, f'{region}_water_levels_admin_cells_{rcp}.dat')
            if os.path.exists(filename):
                fp = np.memmap(filename=filename, dtype='float32', mode='r', shape=water_levels_admin_cells.shape)
            else:
                fp = np.memmap(filename=filename, dtype='float32', mode='w+', shape=water_levels_admin_cells.shape)
                fp[:] = water_levels_admin_cells[:]
                fp.flush()
                
            return fp
        else: return water_levels_admin_cells


    @staticmethod
    def sample_water_level(
        admin_name, 
        gov_admin_idx_cells, 
        dikes_idx_govs,
        coastal_fps_gov,
        dike_heights,
        cells_on_coastline,
        water_levels_admin_cells,
        indice_cell_agent,
        return_periods: np.ndarray,
        fps: int,
        fps_dikes,
        start_year: int,
        current_year: int,
        rcp: str,
        strategy: str,
        beach_width_floodplain: np.array,
        beach_mask: np.array,
        erosion_effect_fps = False,
        dynamic_fps = True,
        gov_admin_idx_subset = None,

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
        # get idx gov for agents
        gov_admin_idx_agents = gov_admin_idx_cells[indice_cell_agent]


        # Interpolate water level for current year
        water_level = np.full(
            (return_periods.size, indice_cell_agent.size), -1, np.float16)
        
        timestep = current_year - start_year
        if rcp == 'control':
            timestep = 0

        # check if we have to iterate over a subset of gov regions or all gov regions in coastal node
        if not gov_admin_idx_subset is None:
            gov_iterable = gov_admin_idx_subset
            water_level = np.full(
                (return_periods.size, indice_cell_agent.size), 0, np.float16)
        else:
            gov_iterable = np.unique(gov_admin_idx_cells)    

        # Fill water levels using polynomials and process fps
        for i, rt in enumerate(return_periods):
            
            # get water level admin cells current year
            water_level_admin_cells = water_levels_admin_cells[i, :, timestep]           

            # iterate over unique gov regions
            for gov_idx in gov_iterable:
                
                # get idx dikes for current gov region
                dikes_idx_gov = dikes_idx_govs[gov_idx]

                # do the following if the gov admin is coastal and has dikes:
                if dikes_idx_gov.size > 0:

                    # get idx agents for current gov region
                    agents_idx_gov = np.where(gov_admin_idx_agents == gov_idx)[0]

                    # get dike heights for current gov region
                    dike_heights_gov = dike_heights[dikes_idx_gov]

                    # get fps for current gov region
                    fps_gov = coastal_fps_gov[gov_idx]
                                
                    # get water levels for dike grid cells
                    water_levels_coastal_cells = water_level_admin_cells[cells_on_coastline][dikes_idx_gov]

                    # using a 0 percent threshold (if 1 dike cell is overtopped, FPS is lost)
                    threshold = 0
                    n_cells_threshold = round(np.max([1, dike_heights_gov.size * threshold]))
                    n_cells_overtopped = np.sum(dike_heights_gov + 0.01 < water_levels_coastal_cells)

                    # for flooding less frequent than fps fill water levels households 
                    if rt >= fps_gov:
                        water_level[i][agents_idx_gov] = np.take(water_level_admin_cells, indice_cell_agent[agents_idx_gov])
                    
                    # # for flooding more frequent than fps check if dikes are overtopped and if so adjust coastal protection standard
                    elif n_cells_overtopped >= n_cells_threshold and rt < fps_gov:
                        if strategy == 'maintain_fps' or rcp == 'control':
                            print(f'WARNING FPS lost in {admin_name} in {rcp} with {strategy}')
                            # raise ValueError(f'FPS lost in {admin_name} in {rcp} with {strategy}')
                            
                        water_level[i][agents_idx_gov] = np.take(water_level_admin_cells, indice_cell_agent[agents_idx_gov])
                        fps_gov = rt
                        coastal_fps_gov[gov_idx] = fps_gov
                        fps_dikes[dikes_idx_gov] = fps_gov
                        # print(f'fps lost in {admin_name}')
                    else:
                        assert rt < fps_gov
                        water_level[i][agents_idx_gov] = np.zeros(agents_idx_gov.size)
                    
                # if gadm region is not coastal (has no dikes), we cannot simulate overtopping (inland gadm has no dike), we only account for current fps
                else:

                    # get idx agents for current gov region
                    agents_idx_gov = np.where(gov_admin_idx_agents == gov_idx)[0]
                    
                    # get fps for current gov region
                    fps_gov = coastal_fps_gov[gov_idx]
                    
                    # for flooding less frequent than fps fill water levels households 
                    if rt >= fps_gov:                   
                        water_level[i][agents_idx_gov] = np.take(water_level_admin_cells, indice_cell_agent[agents_idx_gov])

                    # if protected just fill with zeros
                    else:                   
                        water_level[i][agents_idx_gov] = np.zeros(agents_idx_gov.size)

        # test for weird values 
        assert (water_level >= 0).all()

        return water_level, fps, 0

    @staticmethod
    def calculate_ead_cells_v2(
        coastal_fps_gov,
        gov_admin_idx_cells,
        n_agents: int,
        water_level: np.ndarray,
        dam_func,
        dam_func_dryproof_1m,
        property_value: np.ndarray,
        return_periods: np.ndarray,
        timestep,
        rcp,
        coastal_fps: int,
        
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        '''
        This function is used to calculate the expected annual damages (ead) under no adaptation and when implementing dry flood proofing for each agent

        Args:
            n_agents: number of agents in the current floodplain.
            adapted: an array containing the adaptation status of each agent, with 0 indicating no adaptation, and 1 indicating adaptation.
            water_level: a numpy array containing the inundation levels for each agent associated the different return periods.
            damage_curves: pandas dataframe containing the water levels their associated damage factors under no adaptation, and with implementing dry flood proofing.
            property_value: property value of the agents in the current floodplain.
            return_periods: a numpy array containing the return periods of flood events included in the model run.
            coastal_fps: flood protection standard of the admin region.
            initial_fps: initial flood protection standard of the admin region.

        Returns:
            damages: a numpy array containing the expected damages without adaptation for each flood event and each agent
            damages_dryproof_1m: a numpy array containing the expected damages under dry flood proofing for each flood event and each agent
            ead: a numpy array containing the expected annual damages (ead) for each agent under no adaptation
            ead_dryproof: a numpy array containing the expected annual damages (ead) for each agent under dry flood proofing
            ead_cba: a dictionary containing the ead for the current fps and the initial fps
        '''

        # set timestep to 0 when in control scenario
        if rcp == 'control':
            timestep = 0
        
        # Indicate maximum damage per household
        max_dam = property_value

        # pre-allocate empty array with shape (n_floods, n_agents) for number
        # of damage levels and number of households
        damages = np.full((len(return_periods), n_agents), -1, dtype=np.float32)
        damages_dryproof_1m = np.zeros(
            (len(return_periods), n_agents), dtype=np.float32)

        # make a copy to adjust water levels and not to overwrite original levels
        water_level_copy = water_level[:, :, timestep].copy()

        for i, rt in enumerate(return_periods):  
            water_level_copy[i][np.isnan(water_level_copy[i])] = 0
            water_level_copy[i][water_level_copy[i] < 0] = 0
            water_level_copy[i][water_level_copy[i] > 6] = 6
            
            for gov_idx in np.unique(gov_admin_idx_cells):
                # get cells in current gov region
                cells_idx_gov = np.where(gov_admin_idx_cells == gov_idx)[0]
                # get fps in current gov region
                fps_gov = coastal_fps_gov[gov_idx]

                if rt >= fps_gov:
                    # calculate damage per retun period and store in damage dictory
                    water_level_cm = np.int16(water_level_copy[i]*100)[cells_idx_gov]
                    damages[i][cells_idx_gov] = dam_func[water_level_cm] * max_dam
                    damages_dryproof_1m[i][cells_idx_gov] = dam_func_dryproof_1m[water_level_cm] * max_dam
                else:
                    damages[i][cells_idx_gov] = 0
                    damages_dryproof_1m[i][cells_idx_gov] = 0

        assert (damages != -1).all()
        
        # only take integral over rps not covered by fps
        idx_return_periods = np.where(return_periods >= 0)[0]
        return_periods_adjusted = np.take(return_periods, idx_return_periods, axis = 0)
        x = 1 / return_periods_adjusted

        ead_cells = np.trapz(np.take(damages, idx_return_periods, axis = 0), x, axis=0)
        assert all(ead_cells >= 0) # damages should be equal to or greater than 0

        # Sum and update expected damages per node
        return ead_cells


    @staticmethod
    def calculate_ead_cells_LU(
        coastal_fps_gov,
        gov_admin_idx_cells,
        n_agents: int,
        water_level: np.ndarray,
        dam_func,
        max_dam: np.ndarray,
        return_periods: np.ndarray,
        area_of_grid_cell,
        built_up_area,
        timestep,
        rcp,
        split_by_government = False        
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        '''
        This function is used to calculate the expected annual damages (ead) under no adaptation and when implementing dry flood proofing for each agent

        Args:
            n_agents: number of agents in the current floodplain.
            adapted: an array containing the adaptation status of each agent, with 0 indicating no adaptation, and 1 indicating adaptation.
            water_level: a numpy array containing the inundation levels for each agent associated the different return periods.
            damage_curves: pandas dataframe containing the water levels their associated damage factors under no adaptation, and with implementing dry flood proofing.
            property_value: property value of the agents in the current floodplain.
            return_periods: a numpy array containing the return periods of flood events included in the model run.
            coastal_fps: flood protection standard of the admin region.
            initial_fps: initial flood protection standard of the admin region.

        Returns:
            damages: a numpy array containing the expected damages without adaptation for each flood event and each agent
            damages_dryproof_1m: a numpy array containing the expected damages under dry flood proofing for each flood event and each agent
            ead: a numpy array containing the expected annual damages (ead) for each agent under no adaptation
            ead_dryproof: a numpy array containing the expected annual damages (ead) for each agent under dry flood proofing
            ead_cba: a dictionary containing the ead for the current fps and the initial fps
        '''
        # no slr when in control
        if rcp == 'control':
            timestep = 0

        # Indicate maximum damage per household
        max_dam_cells = max_dam * built_up_area * area_of_grid_cell

        # pre-allocate empty array with shape (n_floods, n_agents) for number
        # of damage levels and number of households
        damages = np.full((len(return_periods), n_agents), 0, dtype=np.float32)

        # make a copy to adjust water levels and not to overwrite original levels
        water_level_copy = water_level[:, :, timestep].copy()
        ead_land_use_summed = 0
        ead_split_by_gov = {}

        for i, rt in enumerate(return_periods):  
            water_level_copy[i][np.isnan(water_level_copy[i])] = 0
            water_level_copy[i][water_level_copy[i] < 0] = 0
            water_level_copy[i][water_level_copy[i] > 6] = 6
            water_level_cm = np.int16(water_level_copy[i]*100)
            damages[i] = dam_func[water_level_cm] * max_dam_cells # only calculate damages for all cells

        for gov_idx in np.unique(gov_admin_idx_cells):
            # get cells in current gov region
            cells_idx_gov = np.where(gov_admin_idx_cells == gov_idx)[0]
                    
            # only take integral over rps not covered by fps
            idx_return_periods = np.where(return_periods >= coastal_fps_gov[gov_idx])[0]
            return_periods_adjusted = np.take(return_periods, idx_return_periods, axis = 0)
            x = 1 / return_periods_adjusted
        
            # subset agents within gov region
            cells_idx_gov = np.where(gov_admin_idx_cells == gov_idx)[0]

            # sum land use damages
            LU_damages_summed = damages[:, cells_idx_gov].sum(axis=1)
        
            # take trapz integral over return periods
            ead_land_use = np.trapz(np.take(LU_damages_summed, idx_return_periods, axis = 0), x).astype(np.int64)
        
            # add the 1/1000 year to this to account for interpolation range [0, 0.001]
            ead_land_use += LU_damages_summed[0] * 1E-3
            ead_split_by_gov[gov_idx] = ead_land_use
            ead_land_use_summed += ead_land_use
        
        if split_by_government:
            return ead_land_use_summed, ead_split_by_gov
        else:
            # Sum and update expected damages per node
            return ead_land_use_summed


    @staticmethod
    def calculate_ead(
        n_agents: int,
        adapted: np.ndarray,
        water_level: np.ndarray,
        dam_func,
        dam_func_dryproof_1m,
        property_value: np.ndarray,
        return_periods: np.ndarray,
        coastal_fps: int,
        initial_fps: int,
        coastal_fps_gov,
        split_by_gov=False,
        gov_admin_idx_cells=None,
        indice_cell_agent=None,
        gradual_fps = True
        
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        '''
        This function is used to calculate the expected annual damages (ead) under no adaptation and when implementing dry flood proofing for each agent

        Args:
            n_agents: number of agents in the current floodplain.
            adapted: an array containing the adaptation status of each agent, with 0 indicating no adaptation, and 1 indicating adaptation.
            water_level: a numpy array containing the inundation levels for each agent associated the different return periods.
            damage_curves: pandas dataframe containing the water levels their associated damage factors under no adaptation, and with implementing dry flood proofing.
            property_value: property value of the agents in the current floodplain.
            return_periods: a numpy array containing the return periods of flood events included in the model run.
            coastal_fps: flood protection standard of the admin region.
            initial_fps: initial flood protection standard of the admin region.

        Returns:
            damages: a numpy array containing the expected damages without adaptation for each flood event and each agent
            damages_dryproof_1m: a numpy array containing the expected damages under dry flood proofing for each flood event and each agent
            ead: a numpy array containing the expected annual damages (ead) for each agent under no adaptation
            ead_dryproof: a numpy array containing the expected annual damages (ead) for each agent under dry flood proofing
            ead_cba: a dictionary containing the ead for the current fps and the initial fps
        '''

        # Indicate maximum damage per household
        max_dam = property_value

        # pre-allocate empty array with shape (n_floods, n_agents) for number
        # of damage levels and number of households
        damages = np.zeros((len(return_periods), n_agents), dtype=np.float32)
        damages_dryproof_1m = np.zeros(
            (len(return_periods), n_agents), dtype=np.float32)

        # make a copy to adjust water levels and not to overwrite original levels
        water_level_copy = water_level.copy()

        for i in range(return_periods.size):  
            water_level_copy[i][water_level_copy[i] < 0] = 0
            water_level_copy[i][water_level_copy[i] > 6] = 6
            # calculate damage per retun period and store in damage dictory
            # place the damage output in the empty array
            # calculate inundation in cm
            water_level_cm = np.int16(water_level_copy[i]*100)
            damages[i] = dam_func[water_level_cm] * max_dam
            damages_dryproof_1m[i] = dam_func_dryproof_1m[water_level_cm] * max_dam

        # calculate ead on damage array along the first axis
        household_damages = np.zeros_like(damages)
        household_damages[:, adapted != 1] = damages[:, adapted != 1]
        household_damages[:, adapted == 1] = damages_dryproof_1m[:, adapted == 1]

        # initialize ead dictionary
        ead_split_by_gov = {}
        ead_summed = 0

        for gov_idx in coastal_fps_gov.keys():
            gov_admin_idx_agents = gov_admin_idx_cells[indice_cell_agent]   
                      
            # only take integral over rps not covered by fps
            idx_return_periods = np.where(return_periods >= coastal_fps_gov[gov_idx])[0]
            return_periods_adjusted = np.take(return_periods, idx_return_periods, axis = 0)
            x = 1 / return_periods_adjusted
            
            # subset agents within gov region
            idx_agents_within_gov = np.where(gov_admin_idx_agents == gov_idx)[0]

            # sum household damages
            household_damages_summed = household_damages[:, idx_agents_within_gov].sum(axis=1)
            
            # take trapz integral over return periods
            ead_households = np.trapz(np.take(household_damages_summed, idx_return_periods, axis = 0), x).astype(np.int64)
            
            # add the 1/1000 year to this to account for interpolation range [0, 0.001]
            ead_households += household_damages_summed[0] * 1E-3
            ead_split_by_gov[gov_idx] = ead_households
            ead_summed += ead_households
        
        if split_by_gov:           
            return damages, damages_dryproof_1m, ead_summed, ead_split_by_gov
        else:
            # calculate ead
            return damages, damages_dryproof_1m, ead_summed

    @staticmethod
    def stochastic_flood(
        random_state,
        current_geom: str,
        water_levels: dict,
        return_periods: np.ndarray,
        flooded: np.ndarray,
        risk_perceptions: np.ndarray,
        flood_timer: np.ndarray,
        risk_perc_min: float,
        risk_perc_max: float,
        risk_decr: float,
        settings: dict,
        fps_dikes: np.ndarray,
        current_year: int,
        spin_up_flag: bool,
        flood_tracker: int,
        verboise=True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        '''
        This function simulates stochastic flooding using a random number generator. In its current
        implementation only one flood event can occur per year.

        Args:
            water_levels: dictionary containing arrays of inundation levels for each agent associated with various return periods.
            return_periods: an array containing the return periods of flood events included in this model.
            flooded: an array containing the flood status of each agent, with 0 indicating not flooded, and 1 indicating flooded.
            risk_perceptions: an array containing the flood risk perceptions of each agent
            method: random if random draw, or a single year.
        Returns:
            flooded: the updated array containing the flood status of each agent
            risk_perceptions: the updated array containing the risk perception of each agent.
            flood_timer: store the year in which the flood event occured
            flood_tracker: store the return period of the simulated flood event'''

        # reset flooded to 0 for all households
        flooded *= 0
        flood_tracker *= 0

        # update flood timer for all households
        flood_timer += 1

        if settings['random_flood'] or spin_up_flag: # allways allow random floods in spinup
            # No flooding in spin up
            if not settings['spin_up_flood'] and spin_up_flag:
                pass

            else:
                # Simulate flooding based on random draw
                random_draw = random_state.random()
                for i, rt in enumerate(return_periods):
                    if random_draw < (1 / rt):
                        flooded[water_levels[i] > 0] = 1
                        # Set flood timer to zero for those who experienced
                        # flooding
                        flood_timer[water_levels[i] > 0] = 0
                        if (rt >= fps_dikes).any():
                            flood_tracker = rt
                        break
        else:

            if isinstance(settings['user_floods'], dict):
                user_floods = settings['user_floods']
            else:
                user_floods = {}

            if current_year in user_floods.keys():
                rt = user_floods[current_year]
                if verboise:
                    print(
                        f'user defined flood in {current_geom} with return period {rt}')

                flooded[water_levels[np.where(return_periods == rt)[
                    0][0]] > 0] = 1
                # Count times people have experienced waterlevels of more than
                # 3 meters (Not used)
                # Set flood timer to zero for those who experienced flooding
                flood_timer[water_levels[np.where(
                    return_periods == rt)[0][0]] > 0] = 0

        risk_perceptions = risk_perc_max * \
            1.6 ** (risk_decr * flood_timer) + risk_perc_min
        return flooded, risk_perceptions, flood_timer, flood_tracker
