import numpy as np
import numba as nb
from agents.base_nodes import HouseholdBaseClass
from decision_module import  gravity_model
from agents.node_properties import InlandNodeProperties


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
        self.beach_width_admin = None
        self.income_percentile = np.array([-9999])
        
        HouseholdBaseClass.__init__(self, model, agents)
        np.random.seed(0)
        self.model.set_seed(0)
        
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
            self.n = np.int(self.population / (self.max_household_size+1/2))       
            if self.n > self.n: raise ValueError("Cannot remove more households than present in the area") 
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
        if (self.model.settings['general']['include_migration'] or self.model.spin_up_flag) and not self.model.calibrate_flag:
            # Use gravity model to determine number of households migrating between regions
            gravity_dictionary = {}
            # Array to store generated household sizes
            household_sizes_all = np.array([], dtype=np.int16)
            n_people_to_move_dict = np.full(len(self.agents.regions.ids), -1, dtype=np.int16)
            n_people_to_move = 0

            # Itterate over each destination node       
      
            # Filter out floodplains
            # Make sure the array is in the correct order
            # assert self.agents.regions.ids[-1].endswith('_flood_plain')

            # Filter
            destinations = [region for region in self.agents.regions.ids if not region.endswith('_flood_plain')]

            # Filter admin
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
                    # extract destination effect
                    destination_effect = self.agents.regions.destination_effects_gravity_model[i]


                    flood_dest = False
                    if dest_flood_plain in self.agents.regions.ids:
                        flood_dest = True
                        dest_flood_plain_idx = self.agents.regions.ids.index(dest_flood_plain)
                        population_flood_plain_dest = self.agents.regions.population[dest_flood_plain_idx]

                    pop_i = self.population
                    pop_j = self.agents.regions.population[i] + population_flood_plain_dest
                    age_i = self.average_age_node
                    age_j = self.agents.regions.average_age_node[i]
                    inc_i = self.income_region
                    inc_j = self.agents.regions.income_region[i]
                    origin_effect = self.origin_effect_gravity_model
                    destination_effect = destination_effect

                    if self.geom_id in self.coastal_admins:
                        coastal_i = 1
                    else:
                        coastal_i = 0
                    
                    if destination in self.coastal_admins:
                        coastal_j = 1
                    else:
                        coastal_j = 0

                    # convert distance to kilometers
                    distance = self.distance_vector[i]
                  
                    # apply gravity model to calculate people flow
                    flow_people = gravity_model(
                        pop_i=pop_i,
                        pop_j=pop_j,
                        inc_i=inc_i,
                        age_i=age_i,
                        age_j=age_j,
                        inc_j=inc_j,
                        coastal_i=coastal_i,
                        coastal_j=coastal_j,
                        distance=distance,
                        origin_effect=origin_effect,
                        destination_effect = destination_effect)
                    

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
            move_to_region_per_household = self.create_household_destinations(move_to_region_per_household, gravity_dictionary)
            # for i in gravity_dictionary.keys():
            #     move_to_region_per_household = np.append(move_to_region_per_household, np.repeat(i, gravity_dictionary[i]))    
        
        
        else:
            n_households_to_move = None

        if not n_households_to_move:  # no household are moving
            return None

        # Create household attributes for move dictionary
        n_movers, to_region, household_id, gender, risk_aversion, age, income_percentile, income, risk_perception = self._generate_households(
            n_households_to_move=n_households_to_move,
            household_sizes=household_sizes_all,
            move_to_region_per_household = move_to_region_per_household,
            hh_risk_aversion = self.model.settings['decisions']['risk_aversion'],
            init_risk_perception = self.model.settings['flood_risk_calculations']['risk_perception']['min'],
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
    
    def create_household_destinations(self, move_to_region_per_household, gravity_dictionary):
        for i in gravity_dictionary.keys():
            move_to_region_per_household = np.append(move_to_region_per_household, np.repeat(i, gravity_dictionary[i]))
        return move_to_region_per_household

    def process_population_change(self):
        population_change, _ = self.ambient_pop_change()
        self.population += population_change
        # Reshuffle households based on population change (improve later, although not relevant)
        self.n = np.int(self.population / (self.max_household_size+1/2))       


    def step(self):
        
        if self.model.settings['general']['include_ambient_pop_change']:
            if self.model.current_time.year > self.model.config['general']['start_time'].year:
                self.process_population_change()
        else:
            pass

