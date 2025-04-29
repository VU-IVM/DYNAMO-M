import numpy as np
from agents.base_nodes import HouseholdBaseClass
from agents.node_properties import InlandNodeProperties


class InlandNode(HouseholdBaseClass, InlandNodeProperties):
    def __init__(
            self,
            model,
            agents,
            geom,
            distance_matrix,
            n_households_per_region,
            idx,
            max_household_size):
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
        
        # get max pop density
        self.max_pop = data.max() * data[data>0].size

        # Average household size
        self.n = self.population // ((max_household_size + 1) / 2)      # calculate the number of households based on the population and the maximum household size
        HouseholdBaseClass.__init__(self, model, agents)                # initiate the household base class
        
        # values not to export:
        self.coastal_fps = None
        self.initial_fps = None
        self.percentage_adapted = None
        self.ead_total = None
        self.ead_residential = None

    def initiate_admin_lists(self):
        '''Creates lists with coastal nodes and inland nodes (for easy reference in move function)'''

        if not hasattr(self, 'coastal_nodes_in_iso3'):
            self.coastal_nodes_in_iso3 = [
                region for region in self.model.agents.regions.ids if
                region.endswith('plain') and region.startswith(self.geom_id[:3])]  
            
            self.coastal_admins_in_iso3 = [coastal_node[:-12] for coastal_node in self.coastal_nodes_in_iso3]
            
            self.inland_nodes_in_iso3 = [
                region for region in self.model.agents.regions.ids if not
                region.endswith('plain') and region.startswith(self.geom_id[:3])]  

    def initiate_agents(self):
        return None

    def add(self, people, households):
        '''Adds people and households to the node.
        
        Args:
            people: A dictionary containing the people moving into the node.
            households: A dictionary containing the households moving into the node.
        
        Returns:
            None'''

        # Initiate admin lists here as well
        self.initiate_admin_lists()

        # Migration
        n_households = np.unique(people['household_id']).size       # get the number of households moving in based on their unique identifyier
        self.n_moved_in_last_timestep = n_households                # add this number to the n_moved_in_last_timestep
        self.n += n_households                                      # add to the total number of households living in the node [n]
        self.population += len(people['to'])                        # add the total number of individuals  moving into the node to the population. 
        assert self.population >= 0

        # set admin to full if max pop is exceeded (only for true inland nodes; we do not know if the coastal node is full yet)
        if self.population > self.max_pop and self.geom_id not in self.coastal_admins_in_iso3:
            self.admin_full = True

    def remove(self, n, n_movers):
        '''Removes people and households from the node.
        
        Args: 
            n: The number of households to remove.'
            n_movers: The number of people to remove.
        
        Returns:    
            None'''

        if n > self.n:                                              # check if the number of households to move out is larger than the number of households present in the node 
            self.n = np.max([n, int(self.population /
                            (self.max_household_size + 1 / 2))])    # if so, reshuffle individuals into households based on the population and the maximum household size
        if n_movers > self.population:                              # check if the number of people to move out is larger than the population present in the node
            raise ValueError(
                f'Cannot remove more people than present in {self.geom_id}')
        self.n_moved_out_last_timestep = n                          # add the number of households moving out to the n_moved_out_last_timestep
        self.people_moved_out_last_timestep = n_movers              # add the number of people moving out to the people_moved_out_last_timestep
        self.n -= n                                                 # subtract the number of households moving out from the total number of households in the node
        self.population -= n_movers                                 # subtract the number of people moving out from the population
        assert self.population >= 0

    def move(self):
        '''Simulates migration from the inland node using the gravity model.
        Flows towards coastal admins are split between the coastal node and inland node based on their population.'''
        assert self.population >= 0
        if not (self.model.settings['agent_behavior']['include_migration']
                or self.model.spin_up_flag) and not self.model.calibrate_flag: # if migration is not included in the model, terminate function here

            return None, None
        
        # else iterate over destinations and calculate flows       
        self.initiate_admin_lists()                                         # create lists with coastal nodes and inland nodes (for easy reference)
        self.n_moved_out_last_timestep = 0
        self.n_moved_in_last_timestep = 0
        region_ids = self.model.agents.regions.ids.copy()
        region_population = self.agents.regions.population_snapshot
        income_regions = tuple(self.agents.regions.income_region)

        gravity_dictionary = {}                                             # dictionary containing flows towards each region idx
        household_sizes_all = np.full(int(self.population), -1, dtype=np.int32)  # array to store the household sizes of households moving out of inland nodes. 
        i_start = 0                                                         # index counter for filling household sizes array

        n_people_to_move_arr = np.full(len(region_ids), 0, dtype=np.int16)  # array containing the number of people to move to each region. 
        n_people_to_move = 0                                                # initiating the total nr of people moving out of the node.
        pop_tracker = int(self.population)                                  # tracker to make sure not people are moving out than present in node. 
        
        pop_i = self.population
        if self.geom_id in self.coastal_admins_in_iso3:
            coastal_i = 1
        else:
            coastal_i = 0
        # iterate over inland nodes as destinations
        for destination in self.inland_nodes_in_iso3:
            if destination == self.geom_id:                         # flows to own region are set to zero.
                continue                                            # and continue loop.
            idx_inland = region_ids.index(destination)              # get the index of the destination node in the list of all households
            pop_j = region_population[idx_inland]                   # get the population residing in the destination node (inland)
            if destination in self.coastal_admins_in_iso3:          # if there also is a floodplain within the destination admin ->
                coastal_id = destination + '_flood_plain'
                idx_coastal = region_ids.index(coastal_id)          # find its index
                if self.agents.regions.admin_full[idx_coastal]:     # check if migration to floodplain is allowed
                    pop_floodplain = 0                              # if not set pop to 0 so no flow emerges
                else:
                    pop_floodplain = region_population[idx_coastal] # get population in floodplain
                pop_j += pop_floodplain                             # and add the population to the population residing in the destination
                coastal_j = 1                                       # set coastal dummy variable
            else:
                pop_floodplain = 0
                coastal_j = 0
            
            # get other gravity params
            inc_i = income_regions[self.admin_idx]                  # take own income region
            inc_j = income_regions[idx_inland]                      # take income inland node
            distance = self.distance_vector[idx_inland]             # take distance to inland node

            # apply gravity model
            if pop_i > 0 and pop_j > 0:                             # only simulate migration from and towards regions with population
                flow_people = self.agents.decision_module.gravity_model(
                    pop_i=pop_i,
                    pop_j=pop_j,
                    distance=distance,
                    region_id=self.UN_region_id,
                    geom_id = self.geom_id)     
            else:
                flow_people = 0

            # make sure not more people than present move.
            if pop_tracker - flow_people <= 0:                      # check if the flow exceeds the population residing in the node. 
                flow_people = np.min([flow_people, pop_tracker])    # if so break loop and stop simulating migration.
                pop_i = 0
                self.model.logger.info(
                    f'no more population in inland node {self.geom_id}'
                    )
            pop_tracker -= flow_people

            # add flow to n_people to move
            n_people_to_move += flow_people

            # if destination is coastal, distribute flow between inland and coastal portion.
            if coastal_j == 1:
                if pop_j > 0:
                    frac_pop_coastal = pop_floodplain/ pop_j   # get fraction of the flow that could be allocated in the coastal floodplain
                    flow_coastal = round(frac_pop_coastal * flow_people)    # split gravity flow between coastal node...

                    if self.model.settings['decisions']['migration']['limit_admin_growth'] and flow_coastal>0:
                        # scale fraction that can be allocated in the floodplain based on available housing                       
                        average_household_size_coastal_j = self.agents.regions.all_households[idx_coastal].size.mean()
                        n_available_housing_coastal_j = self.agents.regions.all_households[idx_coastal].n_available_housing
                        available_housing_people = n_available_housing_coastal_j * average_household_size_coastal_j                      

                        # adjust for frac of total population in pop_i
                        frac_total_pop_in_node = self.population/ self.population_in_country
                        # get the fraction of the flow that could be allocated ()
                        frac_housing_available = np.min([(available_housing_people * frac_total_pop_in_node) / (flow_coastal / frac_total_pop_in_node), 1])
                        # multiply this with the flow
                        flow_coastal = int(flow_coastal * frac_housing_available)

                else:
                    flow_coastal = 0

                flow_inland = round(flow_people - flow_coastal)         # and inland node

                n_people_to_move_arr[idx_inland] = flow_inland            # store the flows in the gravity flow dictionary
                n_people_to_move_arr[idx_coastal] = flow_coastal
                # check if flow does not result in too much coastal population growth
            else: 
                flow_coastal = 0
                flow_inland = flow_people
                n_people_to_move_arr[idx_inland] = flow_inland

            if flow_coastal > 0:
                # get size and household type array from destination 
                hh_size_destination = self.agents.regions.all_household_sizes[idx_coastal]      # get the household sizes in the coastal destination region
                hh_types_destination = self.agents.regions.all_household_types[idx_coastal]     # get the household types in the coastal destination region
                max_household_size = hh_size_destination.max()                                  # get the maximum household size in the destination
                household_sizes = self.return_household_sizes(
                    flow_people=flow_coastal, 
                    max_household_size=max_household_size, 
                    household_sizes_dest=hh_size_destination, 
                    household_types_dest=hh_types_destination)                                  # group people flow in households based on destination household sizes
                assert household_sizes.sum() == flow_coastal                                    # assert that the correct number of households is generated
                household_sizes_all[i_start:household_sizes.size + i_start] = household_sizes   # fill the array containing all household sizes of people moving out
                gravity_dictionary[idx_coastal] = household_sizes.size                          # store the number of households moving out
                i_start += household_sizes.size                                                 # update index

            if flow_inland > 0:
                household_sizes = self.return_household_sizes(
                flow_people=flow_inland,
                max_household_size=self.max_household_size
                )                                                                               # create household sizes for households moving to inland nodes
                assert household_sizes.sum() == flow_inland
                household_sizes_all[i_start:household_sizes.size + i_start] = household_sizes   # fill the array containing all household sizes of people moving out 
                gravity_dictionary[idx_inland] = household_sizes.size                           # store the number of households moving out
                i_start += household_sizes.size                                                 # update index

        # filter out redundancy
        household_sizes_all = household_sizes_all[:i_start]
        n_households_to_move = household_sizes_all.size
        
        # parse to create move to region per household
        move_to_region_per_household = self.create_household_destinations(
            n_households_to_move, gravity_dictionary)
        assert move_to_region_per_household.size == n_households_to_move

        # Create household attributes for move dictionary
        n_movers, to_region, household_id, gender, age, income_percentile, household_type, income, risk_perception, risk_aversion = self._generate_households(
            n_households_to_move=n_households_to_move,
            household_sizes=household_sizes_all,
            move_to_region_per_household=move_to_region_per_household,
            init_risk_aversion=self.model.settings['decisions']['risk_aversion'],
            init_risk_perception=self.model.settings['flood_risk_calculations']['risk_perception']['min'],
        )

        assert n_movers == household_sizes_all.sum()
        if n_movers == 0:
            # If no households are moving, return None
            return None, None


        # Remove housheold and people from the population
        self.remove(n_households_to_move, n_movers)

        # create array with cells to move to (only relevant for moving to coastal node)
        cells_to_move_to = np.full(n_households_to_move, -1, np.int64)

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
            "to": move_to_region_per_household,
            "household_id": np.unique(household_id),
            "risk_aversion": risk_aversion,
            "income_percentile": income_percentile,
            "household_type": household_type,
            "income": income,
            "risk_perception": risk_perception,
            "cells_to_move_to": cells_to_move_to
            }
        
        if self.population > 0:
            self.percentage_moved = np.round(n_movers/ self.population*100, 2)

        return people, households

    @staticmethod
    def create_household_destinations(
            n_households_to_move,
            gravity_dictionary):
        '''Creates an array in which each element represents the destination node of each household moving out of an inland node.
        
        Args:
            n_households_to_move: The number of households moving out of the node.
            gravity_dictionary: A dictionary containing the number of households moving to each node.
        
        Returns:
            move_to_region_per_household: An array containing the destination node of each household moving out of the node.'''

        # make array to fill
        move_to_region_per_household = np.full(
            n_households_to_move, -1, np.int32)
        fill_between = np.array([0, 0])
        for i in gravity_dictionary.keys():
            if gravity_dictionary[i] != 0:
                fill_between[1] += gravity_dictionary[i]
                move_to_region_per_household[fill_between[0]:fill_between[1]] = np.full(
                    gravity_dictionary[i], i)
                fill_between[0] = fill_between[1]

        assert (move_to_region_per_household != -
                1).all()  # check if array is filled
        return move_to_region_per_household

    def process_population_change(self):
        '''Processes population change as projected in the population projection files. It also reshuffles household sizes based on the population change.'''
        population_change, _ = self.ambient_pop_change()
        self.population += population_change
        self.population = np.max([self.population, 0])
        assert self.population >= 0
        # Reshuffle households based on population change (improve later,
        # although not relevant)
        if self.population > 0:
            self.n = self.population
        else:
            self.n = 0

    def step(self):
        if self.model.settings['general']['include_ambient_pop_change']:
            self.process_population_change()
        else:
            pass
