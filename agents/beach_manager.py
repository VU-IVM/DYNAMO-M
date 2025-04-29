from honeybees.agents import AgentBaseClass
import numpy as np
import pandas as pd


class BeachManager(AgentBaseClass):
    '''This class contains the BeachManager agents. For now, they only make decisions on beach renourishment.'''

    def __init__(self, model, agents, beaches):
        """
        Initializes an instance of the BeachManager class.

        Args:
            model: An instance of the model class.
            agents: An instance of the agents class.

        Returns:
        None
        """

        self.model = model
        self.agents = agents
        self.beaches = beaches
        self.coastal_nodes = [
            household for household in self.agents.regions.all_households if household.geom_id.endswith('flood_plain')]
        self.ids = [
            coastal_node.geom_id for coastal_node in self.coastal_nodes]

        self.spendings_admin = np.zeros(len(self.ids))
        self.total_annual_spendings = 0

        #### Beach settings ####
        self.beach_length = 1_000
        self.desired_width = 100
        self.minimal_beach_width = 100 # set to 100 to offsett any erosion within model extent
        self.cost_m3_sand = 7 # set to 1 to only get volume. 
        self.fixed_cost = 0
        self.interval = 1
        self.renourishment_intervals = np.arange(2016, 2081, self.interval)

        # initiate random conditions of time since last renourishment
        self.time_since_renourishment = self.model.random_module.random_state.integers(0, 5, len(self.ids))

        self.strategies = {}
        self.strategies['hold_the_line'] = self.refill_all_populated_beaches_in_admin
        self.strategies['cba'] = self.refill_beaches_CBA

        AgentBaseClass.__init__(self)

    #####################
    ###### Helpers ######
    #####################

    def time_discounting(self, value, horizon, rate):
        time_array = np.arange(0, horizon)
        discounts = 1 / (1 + rate)**time_array
        discounted_value = np.sum(value * discounts)
        return discounted_value

    ########################
    ###### Strategies ######
    ########################

    def refill_all_beach_cells(self):
        assert np.max(self.beaches.beach_width[self.beaches.indices_beach_cells]) <= self.desired_width
        area_to_fill = (
            self.desired_width - self.beaches.beach_width[self.beaches.indices_beach_cells]) * self.beach_length
        volumes_to_fill = area_to_fill * \
            self.beaches.beach_depth_of_closure[self.beaches.indices_beach_cells]
        cost_to_renourish = volumes_to_fill.sum() * self.cost_m3_sand + self.fixed_cost
        self.beaches.beach_width[self.beaches.indices_beach_cells] = self.desired_width
        self.total_annual_spendings = cost_to_renourish

    def refill_all_beach_cells_hold_the_line(self):
        '''This strategy refills all beach cells that are eroding and where the beach width is less than the assigned minimum width. It does not take into account the cost and benefits of the renourishment.'''

        assert np.max(self.beaches.beach_width[self.beaches.indices_beach_cells]) <= self.desired_width
        
        # subset beaches with width smaller than minimum
        # indices to refill
        indices_to_fill = np.where(np.logical_and(self.beaches.beach_width != -1, self.beaches.beach_width < self.minimal_beach_width))

        area_to_fill = (
            self.desired_width - self.beaches.beach_width[indices_to_fill]) * self.beach_length

        volumes_to_fill = area_to_fill * \
            self.beaches.beach_depth_of_closure[indices_to_fill]
        
        cost_to_renourish_cells = volumes_to_fill * self.cost_m3_sand + self.fixed_cost
        cost_to_renourish = cost_to_renourish_cells.sum()

        # fill beaches
        self.beaches.beach_width[indices_to_fill] = np.maximum(self.beaches.beach_width[indices_to_fill], self.desired_width)
        self.total_annual_spendings = cost_to_renourish

        # store times renourished
        self.agents.beaches.cells_times_renourished[indices_to_fill] += 1

        # store gridded costs of renourishment
        self.agents.beaches.cells_renourishment_costs[indices_to_fill] += cost_to_renourish_cells

        assert np.min(self.beaches.beach_width[self.beaches.indices_beach_cells]) >= self.minimal_beach_width
       
    def refill_all_populated_beaches_in_admin(self):
        '''This strategy refills all beaches that are populated by peoplem and are below the desired width. It does not take into account the cost and benefits of the renourishment.'''
        # Iterate over coastal nodes
        self.spendings_admin *= 0
        for i, coastal_node in enumerate(self.coastal_nodes):
            # execute if people are residing in coastal grid cells
            if hasattr(coastal_node, 'indices_admin_in_beach_cells'):

                # cost to fill
                assert (
                    self.beaches.beach_width[coastal_node.indices_admin_in_beach_cells] <= self.desired_width).all()

                area_to_fill = (
                    self.desired_width - self.beaches.beach_width[coastal_node.indices_admin_in_beach_cells]) * self.beach_length
                volumes_to_fill = area_to_fill * \
                    self.beaches.beach_depth_of_closure[coastal_node.indices_admin_in_beach_cells]
                cost_to_renourish_cells = volumes_to_fill * self.cost_m3_sand + self.fixed_cost
                cost_to_renourish = cost_to_renourish_cells.sum()
                self.spendings_admin[i] = cost_to_renourish

                # refill eroding segments
                self.beaches.beach_width[coastal_node.indices_admin_in_beach_cells] = np.maximum(
                    self.beaches.beach_width[coastal_node.indices_admin_in_beach_cells], self.desired_width)

                self.agents.beaches.cells_times_renourished[coastal_node.indices_admin_in_beach_cells] += 1

                # store gridded costs of renourishment
                self.agents.beaches.cells_renourishment_costs[coastal_node.indices_admin_in_beach_cells] += cost_to_renourish_cells
                # print(
                #     f'{coastal_node.indices_admin_in_beach_cells[0].size} cells in {coastal_node.geom_id} renourished for {round(cost_to_renourish*1E-6, 3)} million')
        
        self.total_annual_spendings = self.spendings_admin.sum()


    def refill_beaches_CBA(self, complete_cba=True):
        '''Renourish beaches based on CBA.	This function is applied to calculate the CBA of beach renourishment for each beach segment.
        Args:
            None
        Returns:
            None'''

        for i, coastal_node in enumerate(self.coastal_nodes):
            cost_to_renourish_admin = 0
            # execute if people are residing in coastal grid cells
            if hasattr(coastal_node, 'indices_agents_in_beach_cells'):
                beach_mask = coastal_node.beach_proximity_bool == 1

                indices_agents_in_cells_adjusted = tuple(
                    [coastal_node.indices_agents_in_beach_cells[1][beach_mask],
                     coastal_node.indices_agents_in_beach_cells[0][beach_mask]])

                assert indices_agents_in_cells_adjusted[0].shape[0] == np.sum(
                    beach_mask)

                beach_indices_unique = np.unique(
                    indices_agents_in_cells_adjusted, axis=1)

                # calculate cost to refill for each segment
                area_to_fill = (
                    self.desired_width - self.beaches.beach_width[tuple(beach_indices_unique)]) * self.beach_length
                volumes_to_fill = area_to_fill * \
                    self.beaches.beach_depth_of_closure[tuple(beach_indices_unique)]
                cost_to_renourish = volumes_to_fill * self.cost_m3_sand + self.fixed_cost

                # look up beach cells and assign value
                # only iterate over segments if eroded
                eroding_cells = np.where(volumes_to_fill > 0)[0]
                # print(f'nr_cells eroding in {coastal_node.geom_id}: {eroding_cells.size}')
                for cell_id in eroding_cells:
                    # iterate over beach cells
                    beach_cell = beach_indices_unique[:, cell_id]

                    # assert non empty beach cell is sampled
                    assert self.beaches.beach_width[tuple(beach_cell)] != -1

                    # find all agents in the cell
                    agents_in_cell = np.where(
                        np.logical_and(
                            indices_agents_in_cells_adjusted[0] == beach_cell[0],
                            indices_agents_in_cells_adjusted[1] == beach_cell[1]))

                    if complete_cba:
                        # get erosion rate cell
                        erosion_rate = self.agents.beaches.shoreline_change[tuple(
                            beach_cell)]

                        # get future beach width without renourishment
                        beach_width_future = np.full(
                            self.interval, self.beaches.beach_width[tuple(beach_cell)], np.float32)
                        beach_width_future += np.arange(
                            self.interval) * erosion_rate
                        future_amenity_no_renourishment = np.full(
                            self.interval, -1, np.float32)

                        # get future beach width with renourishment
                        beach_width_future_renourished = np.full(
                            self.interval, self.desired_width, np.float32)
                        beach_width_future_renourished += np.arange(
                            self.interval) * erosion_rate
                        future_amenity_with_renourishment = np.full(
                            self.interval, -1, np.float32)

                        # initiate loop
                        agent_wealth = coastal_node.wealth[beach_mask][agents_in_cell]

                        for year in range(self.interval):
                            # calculate without renourishment
                            amenities_no_renourishment = self.model.coastal_amenities.calculate_beach_amenity(
                                agent_wealth=agent_wealth, beach_width=np.full(
                                    agent_wealth.size, beach_width_future[year]), beach_proximity_bool=np.full(
                                    agent_wealth.size, True))
                            future_amenity_no_renourishment[year] = amenities_no_renourishment.sum(
                            )

                            # calculate with renourishment
                            amenities_with_renourishment = self.model.coastal_amenities.calculate_beach_amenity(
                                agent_wealth=agent_wealth, beach_width=np.full(
                                    agent_wealth.size, beach_width_future_renourished[year]), beach_proximity_bool=np.full(
                                    agent_wealth.size, True))
                            future_amenity_with_renourishment[year] = amenities_with_renourishment.sum(
                            )

                    else:
                        # simply sum the beach amenity in that cell
                        amenities_in_cell = coastal_node.beach_amenity[
                            coastal_node.beach_proximity_bool == 1][agents_in_cell]
                        future_amenity_no_renourishment = amenities_in_cell.sum()

                        # Calculate future amenities in cell for one year
                        agent_wealth = coastal_node.wealth[beach_mask][agents_in_cell]
                        beach_width = np.full(
                            agent_wealth.size, self.desired_width)
                        future_amenities = self.model.coastal_amenities.calculate_beach_amenity(
                            agent_wealth=agent_wealth,
                            beach_width=beach_width,
                            beach_proximity_bool=np.full(
                                agent_wealth.size,
                                True),
                            coastal_amenity_function=self.model.data.coastal_amenity_functions['beach_amenity'])
                        future_amenity_with_renourishment = future_amenities.sum()

                    # Calculate the increase in amenity value through
                    # renourishment
                    amenity_gains = future_amenity_with_renourishment - future_amenity_no_renourishment

                    # tester
                    if complete_cba:
                        assert amenity_gains.size == self.interval

                    # apply time dicounting over horizon renourishment
                    discounted_gain = self.time_discounting(
                        value=amenity_gains, horizon=self.interval, rate=0.04)

                    # get cost of renourishment
                    cost_to_renourish_cell = cost_to_renourish[cell_id]

                    # only renourish if amenity exeeds cost
                    if discounted_gain > cost_to_renourish_cell:
                        self.beaches.beach_width[tuple(
                            beach_cell)] = self.desired_width
                        cost_to_renourish_admin += cost_to_renourish_cell

                        self.agents.beaches.cells_times_renourished[tuple(
                            beach_cell)] += 1
                        self.agents.beaches.cells_renourishment_costs[tuple(beach_cell)] += cost_to_renourish_cell


                self.spendings_admin[i] = cost_to_renourish_admin

        self.total_annual_spendings = self.spendings_admin.sum()

    def step(self):
        pass
        # self.spendings_admin *= 0
        # self.total_annual_spendings = 0
        # # update costs
        # # if not self.model.spin_up_flag and self.cost_m3_sand != 1:
        # #     self.cost_m3_sand *= self.agents.GDP_change.gpd_per_capita_dict[self.model.args.area[0]].loc[self.model.current_time.year][0] # works for single countries (first of area list)
        # #     self.fixed_cost *= self.agents.GDP_change.gpd_per_capita_dict[self.model.args.area[0]].loc[self.model.current_time.year][0] # works for single countries (first of area list)
        # # simple way to refill beaches every x years
        # if self.model.current_time.year in self.renourishment_intervals and self.model.args.beach_manager_strategy != 'none':
        #     self.strategies[self.model.args.beach_manager_strategy]()
        
    @property
    def annual_government_spendings(self):
        return self.spendings_admin
