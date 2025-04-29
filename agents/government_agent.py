from honeybees.agents import AgentBaseClass
import numpy as np
import logging
from hazards.flooding.flood_risk import FloodRisk
from scipy import interpolate
from numba.core.decorators import njit

class GovernmentAgent(AgentBaseClass):
    def __init__(self, model, agents):
        """
        Initializes an instance of the GovenmentAgent class.

        Args:
            model: An instance of the model class.
            agents: An instance of the agents class.

        Returns:
        None
        """       
        self.model = model
        self.agents = agents
        self.return_periods = np.array([key for key in self.model.data.inundation_maps_hist.keys()])
        self.store_strategies()
        self.ids = [coastal_node.geom_id for coastal_node in self.agents.regions.all_households]
        self.fps_spendings = np.zeros(len(self.ids))
        self.fps_spendings_relative_to_gdp = np.zeros(len(self.ids))
        self._initiate_logger()

    def _initiate_logger(self):
        logger = logging.getLogger('government')
        logger.setLevel(logging.getLevelName('INFO'))
        file_handler = logging.FileHandler('government.log', mode='w')
        logger.addHandler(file_handler)
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
        file_handler.setFormatter(formatter)
        self.logger = logger
    
    def interpolate_timer_0(self, datapoints, ead_over_time, years_SLR_included):
        interpolater = interpolate.interp1d(x = datapoints, y = ead_over_time)
        ead_over_time_filled = interpolater(np.arange(years_SLR_included))
        return ead_over_time_filled

    @staticmethod
    @njit
    def get_timesteps_overtopping_numba(
            return_periods,
            years_slr_included,
            fps_mutated,
            water_levels_admin_cells,
            cells_on_coastline,
            model_timestep,
            dike_heights,
            timestep_overtop
    ):
        i = 0
        for idx, rt in enumerate(return_periods):
            # iterate over years in SLR included to check for overtopping
            for timestep in np.arange(years_slr_included):
                water_levels_rt_timestep = water_levels_admin_cells[idx, cells_on_coastline, model_timestep+timestep]
                # using a 10 percent threshold (if > 10 percent of dike cells is overtopped, FPS is lost) SHOULD MATCH WITH FLOOD RISK FUNCTION
                threshold = 0.1
                n_cells_overtopped = np.sum(dike_heights < water_levels_rt_timestep)
                if n_cells_overtopped > threshold * dike_heights.size and rt < fps_mutated:
                    if timestep < 2:
                        timestep_overtop[i] = 0
                        timestep_overtop[i+1] = timestep+2
                        i += 2
                    else:
                        timesteps = np.arange(timestep-2, timestep+2)
                        timesteps = np.maximum(timesteps, 0)
                        assert np.sum(timestep_overtop[i: i+timesteps.size] != 0) == 0
                        timestep_overtop[i: i+timesteps.size] = timesteps
                        i += timesteps.size
                    fps_mutated = rt
                    break

    def get_timesteps_overtopping(
        self,
        coastal_node,   
        years_SLR_included, 
        fps,
        dike_heights):

        fps_mutated = int(fps)

        # get idx fps
        # timestep_overtop = np.full(self.return_periods.size+2, 0, np.int32)
        timestep_overtop = np.full(self.return_periods.size * 10, 0, np.int32)
        timestep_overtop[-1] = years_SLR_included
        
        # get timesteps of overtopping
        self.get_timesteps_overtopping_numba(
            self.return_periods,
            years_SLR_included,
            fps_mutated,
            coastal_node.water_levels_admin_cells,
            coastal_node.cells_on_coastline,
            self.model.timestep,
            dike_heights,
            timestep_overtop
        )

        # get timesteps to interpolate 
        datapoints = np.unique(np.minimum(timestep_overtop, 100))
        return datapoints
  
    def interpolate_future_risk_elevated_dike_height(
            self,
            coastal_node,
            decision_horizon,
            datapoints,
            upgrade=True,
        ):

        years_SLR_included = decision_horizon
        if self.model.args.rcp == 'control':
            datapoints = np.array([0, years_SLR_included])
        ead_over_time = {}

        # make some copies to avoid changing the original values
        coastal_fps_gov = coastal_node.coastal_fps_gov.copy()
        dike_heights = coastal_node.dike_heights.copy()
        fps_dikes = coastal_node.coastal_fps_dikes.copy()

        # iterate over gov_idx in coastal node
        populated_gov_idxs = np.unique(coastal_node.gov_admin_idx_cells[coastal_node.indice_cell_agent])
        populated_gov_idxs_with_dikes = np.intersect1d(populated_gov_idxs, coastal_node.gov_admin_idx_cells[coastal_node.cells_on_coastline])

        # elevate dikes to target fps in each gov region
        for gov_idx in populated_gov_idxs_with_dikes:
            # get closest fps for each gov region
            fps_gov = coastal_node.coastal_fps_gov[gov_idx]
            # get target fps
            return_period_idx = np.where(self.return_periods == fps_gov)[0][0]
            if upgrade:
                if coastal_fps_gov[gov_idx] != 1_000:
                    return_period_idx_target = return_period_idx - 1
                    fps_target = self.return_periods[return_period_idx_target]
                else:
                    return_period_idx_target = return_period_idx
                    fps_target = 1_000
            else:
                return_period_idx_target = return_period_idx
                fps_target = fps_gov

            # get required dike heights
            dike_idx = coastal_node.dikes_idx_gov[gov_idx]
            elevated_dike_heights = coastal_node.water_levels_admin_cells[return_period_idx_target, coastal_node.cells_on_coastline[dike_idx], (self.model.timestep+decision_horizon)]
            
            # adjust input for damage calculations
            coastal_fps_gov[gov_idx] = fps_target
            fps_dikes[dike_idx] = fps_target
            dike_heights[dike_idx] = elevated_dike_heights
            
        # calculate future risk for current dike height
        for i, year in enumerate(datapoints):
            water_level, fps, _ = FloodRisk.sample_water_level(
                admin_name=coastal_node.geom_id,
                gov_admin_idx_cells=coastal_node.gov_admin_idx_cells,
                dikes_idx_govs=coastal_node.dikes_idx_gov,
                coastal_fps_gov=coastal_fps_gov,
                dike_heights=dike_heights,
                cells_on_coastline=coastal_node.cells_on_coastline,
                water_levels_admin_cells=coastal_node.water_levels_admin_cells,
                indice_cell_agent = coastal_node.indice_cell_agent,
                return_periods=np.array([key for key in self.model.data.inundation_maps_hist.keys()]),
                fps=coastal_node.coastal_fps,
                fps_dikes=fps_dikes,
                strategy=self.model.args.government_strategy,
                start_year = self.model.config['general']['start_time'].year,
                current_year=self.model.current_time.year+i,
                rcp=self.model.args.rcp,
                beach_width_floodplain=np.array([]), 
                beach_mask=np.array([]), 
                erosion_effect_fps=None
            )

            damages, damages_dryproof_1m, ead_total_fps, ead_split_by_gov = FloodRisk.calculate_ead(
                n_agents=coastal_node.n,
                adapted = coastal_node.adapt,
                water_level=water_level,
                dam_func=self.model.data.dam_func[coastal_node.UN_region_id],
                dam_func_dryproof_1m=self.model.data.dam_func_dryproof_1m[coastal_node.UN_region_id],
                property_value=coastal_node.property_value,
                return_periods=np.array([key for key in self.model.data.inundation_maps_hist.keys()]),
                coastal_fps=fps,
                initial_fps = coastal_node.initial_fps,
                coastal_fps_gov=coastal_fps_gov,
                split_by_gov=True,
                gov_admin_idx_cells=coastal_node.gov_admin_idx_cells,
                indice_cell_agent=coastal_node.indice_cell_agent,)
            
            # check if fps is overtopped
            # if fps < coastal_node.coastal_fps:
            overtopped = True

            # iterate over gov regions and store ead
            for gov_idx in ead_split_by_gov.keys():
                if not gov_idx in ead_over_time.keys():
                    ead_over_time[gov_idx] = np.full(datapoints.size, -1, np.float64)
                    ead_over_time[gov_idx][i] = ead_split_by_gov[gov_idx]
                else:
                    ead_over_time[gov_idx][i] = ead_split_by_gov[gov_idx]

        ead_over_time_filled_dict = {}
        
        # get GDP growth
        GDP_change = self.agents.GDP_change.gpd_per_capita_dict[coastal_node.geom_id[:3]]
        years_included = np.arange(self.model.current_time.year, self.model.current_time.year+years_SLR_included)
        # cap to max year in GDP projections
        years_included = np.minimum(years_included, GDP_change.index[-1])
        GDP_change_values = np.array(GDP_change.loc[years_included]['perc_increase'])

        for gov_idx in ead_over_time.keys():
            ead_over_time_filled_dict[gov_idx] = self.interpolate_timer_0(datapoints, ead_over_time[gov_idx], years_SLR_included)
            ead_over_time_filled_dict[gov_idx] *= GDP_change_values.cumprod()

        return ead_over_time_filled_dict, overtopped

    def interpolate_future_risk_current_dike_height(
            self,
            coastal_node,
            decision_horizon,
            datapoints,
        ):

        years_SLR_included = decision_horizon
        if self.model.args.rcp == 'control':
            datapoints = np.array([0, years_SLR_included])

        ead_over_time = {}

        # make some copies to avoid changing the original values
        coastal_fps_gov = coastal_node.coastal_fps_gov.copy()
        dike_heights = coastal_node.dike_heights.copy()
        fps_dikes = coastal_node.coastal_fps_dikes.copy()

        # calculate future risk for current dike height
        for i, year in enumerate(datapoints):
            water_level, fps, _ = FloodRisk.sample_water_level(
                admin_name=coastal_node.geom_id,
                gov_admin_idx_cells=coastal_node.gov_admin_idx_cells,
                dikes_idx_govs=coastal_node.dikes_idx_gov,
                coastal_fps_gov=coastal_fps_gov,
                dike_heights=dike_heights,
                cells_on_coastline=coastal_node.cells_on_coastline,
                water_levels_admin_cells=coastal_node.water_levels_admin_cells,
                indice_cell_agent = coastal_node.indice_cell_agent,
                return_periods=np.array([key for key in self.model.data.inundation_maps_hist.keys()]),
                fps=coastal_node.coastal_fps,
                fps_dikes=fps_dikes,
                strategy=self.model.args.government_strategy,
                start_year = self.model.config['general']['start_time'].year,
                current_year=self.model.current_time.year+year,
                rcp=self.model.args.rcp,
                beach_width_floodplain=np.array([]), 
                beach_mask=np.array([]), 
                erosion_effect_fps=None
            )

            damages, damages_dryproof_1m, ead_total_fps, ead_split_by_gov = FloodRisk.calculate_ead(
                n_agents=coastal_node.n,
                adapted = coastal_node.adapt,
                water_level=water_level,
                dam_func=self.model.data.dam_func[coastal_node.UN_region_id],
                dam_func_dryproof_1m=self.model.data.dam_func_dryproof_1m[coastal_node.UN_region_id],
                property_value=coastal_node.property_value,
                return_periods=np.array([key for key in self.model.data.inundation_maps_hist.keys()]),
                coastal_fps=fps,
                initial_fps = coastal_node.initial_fps,
                split_by_gov=True,
                coastal_fps_gov=coastal_fps_gov,
                gov_admin_idx_cells=coastal_node.gov_admin_idx_cells,
                indice_cell_agent=coastal_node.indice_cell_agent,)
                
            if self.model.args.rcp == 'control':
                timestep = 0
            else:
                timestep = (self.model.current_time.year + year) - self.model.config['general']['start_time'].year 

            ead_commercial, ead_commercial_split_by_government = FloodRisk.calculate_ead_cells_LU(
                coastal_fps_gov=coastal_fps_gov,
                gov_admin_idx_cells=coastal_node.gov_admin_idx_cells,
                n_agents=coastal_node.locations_admin_cells.shape[0],
                water_level=coastal_node.water_levels_admin_cells,
                dam_func=self.model.data.dam_func_commercial[coastal_node.UN_region_id],
                max_dam=coastal_node.max_damage_commercial,
                return_periods=np.array([key for key in self.model.data.inundation_maps_hist.keys()]),
                area_of_grid_cell=0.15,
                built_up_area = coastal_node.built_up_area,
                timestep=timestep,
                rcp=self.model.args.rcp,
                split_by_government=True
            )       

            ead_industrial, ead_industrial_split_by_government = FloodRisk.calculate_ead_cells_LU(
                coastal_fps_gov=coastal_fps_gov,
                gov_admin_idx_cells=coastal_node.gov_admin_idx_cells,
                n_agents=coastal_node.locations_admin_cells.shape[0],
                water_level=coastal_node.water_levels_admin_cells,
                dam_func=self.model.data.dam_func_industrial[coastal_node.UN_region_id],
                max_dam=coastal_node.max_damage_industrial,
                return_periods=np.array([key for key in self.model.data.inundation_maps_hist.keys()]),
                area_of_grid_cell=0.10,
                built_up_area = coastal_node.built_up_area,
                timestep=timestep,
                rcp=self.model.args.rcp,
                split_by_government=True
            ) 

            # check if fps is overtopped
            # if fps < coastal_node.coastal_fps:
            # iterate over gov regions and store ead
            for gov_idx in ead_split_by_gov.keys():
                if not gov_idx in ead_over_time.keys():
                    ead_over_time[gov_idx] = np.full(datapoints.size, -1, np.float64)
                    ead_over_time[gov_idx][i] = ead_split_by_gov[gov_idx] + ead_commercial_split_by_government[gov_idx] + ead_industrial_split_by_government[gov_idx]
                else:
                    ead_over_time[gov_idx][i] = ead_split_by_gov[gov_idx] + ead_commercial_split_by_government[gov_idx] + ead_industrial_split_by_government[gov_idx]
        
        ead_over_time_filled_dict = {}
        
        # get GDP growth
        GDP_change = self.agents.GDP_change.gpd_per_capita_dict[coastal_node.geom_id[:3]]
        years_included = np.arange(self.model.current_time.year, self.model.current_time.year+years_SLR_included)
        # cap to max year in GDP projections
        years_included = np.minimum(years_included, GDP_change.index[-1])
        GDP_change_values = np.array(GDP_change.loc[years_included]['perc_increase'])

        for gov_idx in ead_over_time.keys():
            ead_over_time_filled_dict[gov_idx] = self.interpolate_timer_0(datapoints, ead_over_time[gov_idx], years_SLR_included)
            ead_over_time_filled_dict[gov_idx] *= GDP_change_values.cumprod()

        # check current fps was overtopped within horizon
        if (fps_dikes != coastal_node.coastal_fps_dikes).any():
            overtopped = True
        else:
            overtopped = False              

        return ead_over_time_filled_dict, overtopped

    def calculate_NPV(
        self,
        coastal_node,
        return_period_idx,
        ead_reduction,
        gov_idx,
        discount_rate,
        decision_horizon,
        cost_m_km,
        maintenance_costs_km,
        ):

        # subset dikes
        dike_idx = coastal_node.dikes_idx_gov[gov_idx]
        
        # get required dike height elevation
        elevated_dike_heights = coastal_node.water_levels_admin_cells[return_period_idx , coastal_node.cells_on_coastline[dike_idx], (self.model.timestep+decision_horizon)]
        current_dike_heights = coastal_node.dike_heights[dike_idx]
        dike_elev = np.maximum((elevated_dike_heights - current_dike_heights), 0)

        # get costs
        total_costs = np.sum(dike_elev * cost_m_km * 1E-3 * coastal_node.coastal_dike_length[dike_idx]) # sum

        # initiate time discount factor array
        discounts = 1 / (1 + discount_rate)**np.arange(decision_horizon)
       

        NPV_arr = np.maximum(ead_reduction * discounts, 0)
        cost_arr = np.zeros(decision_horizon, np.float64)
        cost_arr[0] = total_costs

        # calculate maintenance costs for current and elevated dike lenghts
        total_dike_lenght_current = (coastal_node.coastal_dike_length[dike_idx][current_dike_heights > 0]).astype(np.float64).sum()
        total_dike_length_elevated = (coastal_node.coastal_dike_length[dike_idx][elevated_dike_heights > 0]).astype(np.float64).sum()

        current_maintenance_costs = total_dike_lenght_current * 1E-3 * maintenance_costs_km
        elevated_maintenance_costs = total_dike_length_elevated * 1E-3 * maintenance_costs_km
        
        # calculate difference in maintenance costs
        difference_cost = elevated_maintenance_costs - current_maintenance_costs
        
        # add maintenance costs to cost array
        cost_arr[1:] = difference_cost  # TODO: implement economic growth also in future maintanance costs
        cost_arr *= discounts

        NPV_arr -= cost_arr
        NPV = np.sum(NPV_arr)

        # add maintenance costs to total costs
        # total_costs += total_costs * 0.01 * (decision_horizon-1)

        return NPV, total_costs, elevated_maintenance_costs

    def CBA_upgrade_FPS(
        self,
        coastal_node,
        discount_rate = 0.04, #0.04 in TH2020,
        decision_horizon = 100,
        ):

        '''This function restores the dike height to the target fps when the NPV of the dike height is positive.
        The NPV is calculated as the sum of the discounted EAD reduction minus the costs of raising the dike height.'''

        # get costs
        cost_m_km = coastal_node.dike_elevation_cost
        maintenance_costs_km = coastal_node.dike_maintenance_cost

        # reset spendings
        self.fps_spendings *= 0

        # initiate datapoints for interpolation
        datapoints = np.full(19, -1, np.int16)
        datapoints[:10] = np.arange(0, 20, 2)
        datapoints[10:] = np.arange(20, decision_horizon+1, 10)


        risk_current_dike_height_dict, overtopped = self.interpolate_future_risk_current_dike_height(
            coastal_node,
            decision_horizon,
            datapoints,
        )

        if overtopped:
            risk_maintain_fps_dict, _ = self.interpolate_future_risk_elevated_dike_height(
                coastal_node,
                decision_horizon,
                datapoints,
                upgrade=False
            )
        else: 
            risk_maintain_fps_dict = risk_current_dike_height_dict

        risk_elevated_dike_height_dict, _ = self.interpolate_future_risk_elevated_dike_height(
            coastal_node,
            decision_horizon,
            datapoints,
            upgrade=True
        )

        # iterate over gov_idx in coastal node
        populated_gov_idxs = np.unique(coastal_node.gov_admin_idx_cells[coastal_node.indice_cell_agent])
        populated_gov_idxs_with_dikes = np.intersect1d(populated_gov_idxs, coastal_node.gov_admin_idx_cells[coastal_node.cells_on_coastline])

        for gov_idx in populated_gov_idxs_with_dikes:
            # subset dikes
            dike_idx = coastal_node.dikes_idx_gov[gov_idx]
            
            # get fps
            fps_gov = coastal_node.coastal_fps_gov[gov_idx]

            # get target fps
            return_period_idx_raise = np.where(self.return_periods == fps_gov)[0][0]
            if coastal_node.coastal_fps_gov[gov_idx] != 1_000:
                return_period_idx_raise -= 1
                fps_target = self.return_periods[return_period_idx_raise]
            else:
                fps_target = 1_000
    
            NPV_raise_fps, total_costs_raise_fps, annual_maintenance_costs_raise_fps = self.calculate_NPV(
                coastal_node=coastal_node,
                return_period_idx=return_period_idx_raise,
                ead_reduction=risk_current_dike_height_dict[gov_idx] - risk_elevated_dike_height_dict[gov_idx],
                gov_idx=gov_idx,
                discount_rate=discount_rate,
                decision_horizon=decision_horizon,
                cost_m_km=cost_m_km,
                maintenance_costs_km=maintenance_costs_km,
            )

            # get idx of current fps 
            return_period_idx_current = np.where(self.return_periods == fps_gov)[0][0]
            NPV_maintain_fps, total_costs_maintain_fps, annual_maintenance_costs_maintain_fps = self.calculate_NPV(
                coastal_node=coastal_node,
                return_period_idx=return_period_idx_current,
                ead_reduction=risk_current_dike_height_dict[gov_idx] - risk_maintain_fps_dict[gov_idx],
                gov_idx=gov_idx,
                discount_rate=discount_rate,
                decision_horizon=decision_horizon,
                cost_m_km=cost_m_km,
                maintenance_costs_km=maintenance_costs_km,
            )

            # compare NPV if NPV is positive
            if NPV_raise_fps > 0 or NPV_maintain_fps > 0:
                if NPV_raise_fps > NPV_maintain_fps:
                    # get required dike height elevation
                    elevated_dike_heights = coastal_node.water_levels_admin_cells[return_period_idx_raise , coastal_node.cells_on_coastline[dike_idx], (self.model.timestep+decision_horizon)]                   
                    # checks some things
                    assert coastal_node.coastal_fps_gov[gov_idx] <= fps_target
                    if not all(elevated_dike_heights >= coastal_node.dike_heights[dike_idx]):
                        print(f'WARNING: dike height not elevated but lowered in {coastal_node.geom_id} under {self.model.args.rcp}')
                    # elevate
                    coastal_node.coastal_fps_gov[gov_idx] = fps_target
                    coastal_node.coastal_fps_dikes[dike_idx] = fps_target
                    coastal_node.dike_heights[dike_idx] = elevated_dike_heights
                    self.logger.info(f'{self.model.current_time.year}: FPS of {gov_idx} heightend in {coastal_node.geom_id}')
                    assert coastal_node.geom_id == self.ids[coastal_node.admin_idx]
                    self.fps_spendings[coastal_node.admin_idx] += total_costs_raise_fps 
                    if not self.agents.GDP_change.GDP_country[coastal_node.geom_id[:3]].loc[self.model.current_time.year] == 0:
                        self.fps_spendings_relative_to_gdp[coastal_node.admin_idx] += total_costs_raise_fps / self.agents.GDP_change.GDP_country[coastal_node.geom_id[:3]].loc[self.model.current_time.year]
                    coastal_node.dike_maintenance_costs[gov_idx] = annual_maintenance_costs_raise_fps
                    print(f'FPS of {gov_idx} heightend in {coastal_node.geom_id} to {fps_target}')
                else:
                    # get required dike height elevation
                    elevated_dike_heights = coastal_node.water_levels_admin_cells[return_period_idx_current , coastal_node.cells_on_coastline[dike_idx], (self.model.timestep+decision_horizon)]
                    # check and elevate dikes
                    if not all(elevated_dike_heights >= coastal_node.dike_heights[dike_idx]):
                        print(f'WARNING: dike height not elevated but lowered in {coastal_node.geom_id} under {self.model.args.rcp}')
                    coastal_node.dike_heights[dike_idx] = elevated_dike_heights
                    self.logger.info(f'{self.model.current_time.year}: FPS of {gov_idx} maintained in {coastal_node.geom_id}')
                    assert coastal_node.geom_id == self.ids[coastal_node.admin_idx]
                    self.fps_spendings[coastal_node.admin_idx] += total_costs_maintain_fps     
                    if not self.agents.GDP_change.GDP_country[coastal_node.geom_id[:3]].loc[self.model.current_time.year] == 0:
                        self.fps_spendings_relative_to_gdp[coastal_node.admin_idx] += total_costs_maintain_fps / self.agents.GDP_change.GDP_country[coastal_node.geom_id[:3]].loc[self.model.current_time.year]
                    coastal_node.dike_maintenance_costs[gov_idx] = annual_maintenance_costs_maintain_fps
                    print(f'FPS of {gov_idx} maintained in {coastal_node.geom_id} at {fps_gov}')
            else:
                self.logger.info(f'{self.model.current_time.year}: No action taken in {coastal_node.geom_id} {gov_idx}. Only maintainance costs incurred.')
                self.fps_spendings[coastal_node.admin_idx] += coastal_node.dike_maintenance_costs[gov_idx]

    def _maintain_fps(self, coastal_node):
        '''This function increases dike height to maintain the initial fps'''
       # iterate over gov_idx in coastal node
        gov_idxs_with_dikes = np.unique(coastal_node.gov_admin_idx_cells[coastal_node.cells_on_coastline])

        for gov_idx in gov_idxs_with_dikes:
            # subset dikes
            dike_idx = coastal_node.dikes_idx_gov[gov_idx]
            
            # get fps
            fps_gov = coastal_node.coastal_fps_gov[gov_idx]

            # get target fps
            return_period_idx = np.where(self.return_periods == fps_gov)[0][0]
        
            # get required dike height
            elevated_dike_heights = coastal_node.water_levels_admin_cells[return_period_idx , coastal_node.cells_on_coastline[dike_idx], (self.model.timestep+5)]                   

            # calculate required dike elevation
            dike_elev = np.maximum((elevated_dike_heights - coastal_node.dike_heights[dike_idx]), 0)
        
            # get costs
            cost_m_km = coastal_node.dike_elevation_cost
            total_costs = np.sum(dike_elev * cost_m_km * 1E-3 * coastal_node.coastal_dike_length[dike_idx]) # sum
            # add maintenance costs
            dike_length = coastal_node.coastal_dike_length[dike_idx][elevated_dike_heights > 0].astype(np.float64).sum()
            maintainance_costs = coastal_node.dike_maintenance_cost * dike_length * 1E-3
            # set dike height
            coastal_node.dike_heights[dike_idx] = elevated_dike_heights

            # store costs
            assert coastal_node.geom_id == self.ids[coastal_node.admin_idx]
            self.fps_spendings[coastal_node.admin_idx] += total_costs + maintainance_costs
            if not self.agents.GDP_change.GDP_country[coastal_node.geom_id[:3]].loc[self.model.current_time.year] == 0:
                self.fps_spendings_relative_to_gdp[coastal_node.admin_idx] += total_costs / self.agents.GDP_change.GDP_country[coastal_node.geom_id[:3]].loc[self.model.current_time.year]


    def policy_cycles(self, coastal_node, upgrade = True, interval = 6):
        # initiate array (ugly)
        time_array = np.arange(1, 120, interval)
        flood_status = coastal_node.flood_tracker != 0 
        # if True:
        if ((self.model.timestep in time_array or flood_status) and not self.model.spin_up_flag):# or (self.model.spin_up_cycle in time_array):
            # update dike height when in time array (all nodes updated in same timestep)
            if upgrade:
                self.CBA_upgrade_FPS(coastal_node)
        else:
            coastal_node.CBA_ratio_maintain = np.nan
            coastal_node.CBA_ratio_upgrade = np.nan
    
    def subsidize_adaptation_costs(self, coastal_node):
        if not self.model.spin_up_flag:
            # check which households are unable to invest in dry floodproofing
            constrained_households =  np.where(coastal_node.income * self.model.settings['adaptation']['expenditure_cap'] <= coastal_node.adaptation_costs)
            coastal_node.adaptation_costs[constrained_households] //= 2

    ################################################
    ##########  Government strategies ##############
    ################################################
    def maintain_fps(self, coastal_node):
        self._maintain_fps(coastal_node)      

    def proactive_government(self, coastal_node):
        # iterate over coastal nodes and calculate benefits maintaining current fps
        self.policy_cycles(coastal_node)
    
    def reactive_government(self, coastal_node):
        if coastal_node.flood_tracker != 0 and coastal_node.flood_tracker < coastal_node.initial_fps:
            self.CBA_upgrade_FPS(coastal_node)
    
    def no_adaptation(self, _):
        pass

    def no_government(self, _):
        pass

    def store_strategies(self):
        self.government_strategies = {
            'maintain_fps': self.maintain_fps,
            'proactive_government': self.proactive_government,
            'reactive_government': self.reactive_government,
            'no_adaptation': self.no_adaptation,
            'no_government': self.no_government,
            }

    #################################################
    def step(self, coastal_node):
        if 'government_strategy' in self.model.args:
            strategy = self.model.args.government_strategy
        else:
            strategy = self.model.settings['adaptation']['government_strategy']
        
        if coastal_node.geom_id.endswith('flood_plain'):
            # reset government spendings
            self.fps_spendings[coastal_node.admin_idx] = sum(coastal_node.dike_maintenance_costs.values())
            self.fps_spendings_relative_to_gdp[coastal_node.admin_idx] = 0
            # execute strategy
            self.government_strategies[strategy](coastal_node)
            if self.model.settings['subsidies']['adaptation']:
                self.subsidize_adaptation_costs(coastal_node)