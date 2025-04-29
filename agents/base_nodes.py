import numpy as np
from numba import njit
from honeybees.agents import AgentBaseClass

class HouseholdBaseClass(AgentBaseClass):
    """This class contains all the attributes, properties and functions that are shared by both the coastal and inland agent classes."""

    def __init__(self, model, agents):
        self.model = model
        self.agents = agents

        # Find admin
        self.geom_id = self.geom['properties']['id']

        # assign UN M49 region identifier
        iso3_code = self.geom_id[:3]
        
        if iso3_code == 'TWN': # Taiwan
            self.UN_region_id = 'Asia'
        elif iso3_code == 'XAD': # Akrotiri en Dhekelia
            self.UN_region_id = 'Europe'
        else:
            # look up in table
            if iso3_code not in self.model.data.UNSD_M49.index:
                raise ValueError(f'{iso3_code} not in UN classification scheme.')
            data = self.model.data.UNSD_M49.loc[iso3_code]
            if data['Region Name'] == 'Americas':
                if data['Sub-region Name'] == 'Northern America':
                    self.UN_region_id = data['Sub-region Name'] 
                elif data['Intermediate Region Name'] == 'Caribbean':
                    self.UN_region_id = 'Central America'
                else:
                    self.UN_region_id = data['Intermediate Region Name']
            else:
                self.UN_region_id = data['Region Name'] 

        # load income from map
        # sample from flopros and assign
        data = self.model.data.hh_income_map.sample_geom(self.geom)
        data = data.ravel()
        data = data[data != -1]
        income_region = 0 # initiate
        
        # average_fps in cells
        if data.size > 0:
            income_region = int(np.mean(data)) # take the mean of all cells (gadm1 does not always overlap with GDL regions)
        if data.size == 0 or income_region == 0:
            # if in index get income region from table
            if self.geom_id[:3] in self.model.data.hh_income_table.index:
                income_region = self.model.data.hh_income_table.loc[self.geom_id[:3]]['2015 [YR2015]']
                # transform 2015 USD to 2015 EUR
                income_region *= 0.9015
            else:
                income_region = self.model.data.hh_income_table['2015 [YR2015]'].mean() * 0.9015
                self.model.logger.info(f'no income found for {self.geom_id[:3]}')

        self.income_region = int(income_region * 0.937)# transform income to disposable income (based on OECD data)
      
       # Create income distribution for each region (n=5_000)
        # income in GDL regions in average income, not median.
        # get ratio
        if self.geom_id[:3] in self.model.data.UN_WIID.index:
            mean_median_inc_ratio = self.model.data.UN_WIID.loc[self.geom_id[:3]]['mean_median_ratio']
        else:
            mean_median_inc_ratio = self.model.settings['adaptation']['mean_median_inc_ratio']
        median_income = self.income_region/ mean_median_inc_ratio
        # mean_income = self.income_region * \
        #     self.model.settings['adaptation']['mean_median_inc_ratio']
        mu = np.log(median_income)
        sd = np.sqrt(2 * np.log(self.income_region / median_income))
        self.income_distribution_region = np.sort(
            self.model.random_module.random_state.lognormal(
                mu, sd, 5_000).astype(
                np.int32))  # initiate with 2_000
        self.average_household_income = int(
            self.income_distribution_region.mean())

        # initiate flag allowing inmigration
        self.admin_full = False

        # Initiate the percentage of households implementing dry proofing for
        # all regions
        self.percentage_adapted = None
        self.percentage_insured = None
        self.n_households_adapted = None
        self.perc_people_moved_out = 0
        self.flood_tracker = 0
        self.segment_IDs_admin = None

        self.summed_beach_amenity = 0
        self.beach_amenity_dict = {}
        # Initiate expected damages for all regions
        self.ead_total = 0
        self.ead_residential = 0
        self.people_near_beach = 0
        self.households_near_beach = 0

        self.n_moved_out_last_timestep = 0
        self.n_moved_in_last_timestep = 0
        self.people_moved_out_last_timestep = 0

        self.initiate_agents()

        # Extract region name of floodplain and corresponding inland regions
        # (should be moved to initiate agents)
        coastal_admins = [region['properties']['id'] for region in self.model.area.geoms['admin']
                          if region['properties']['id'].endswith('flood_plain')]

        # List comprehension crashes
        for index in range(len(coastal_admins)):
            coastal_admins.append(coastal_admins[index][:-12])
        self.coastal_admins = coastal_admins

        # Average amenity value in all regions (coastal will be overwritten).
        # Improve this
        self.average_amenity_value = 0
        self.percentage_moved = 0
    
    def select_regions_where_to_move(self):
        # select only regions within the same country (internal migration only)
        # get iso3 of own region 
        iso3_region = self.geom_id[:3]
        regions_select = []
        i = 0
        for region, admin_full in zip(self.agents.regions.ids, self.agents.regions.admin_full):
            if region.startswith(iso3_region) and region != self.geom_id and not admin_full: # and not region.endswith('flood_plain'):
                regions_select.append(i)
            i+=1

        regions_select = np.array(regions_select)
        return regions_select

    @staticmethod
    @njit(cache=True)
    def return_household_sizes(
        flow_people,
        max_household_size,
        household_sizes_dest=np.array([]),
        household_types_dest=np.array([])
        ):
        '''Small helper function to speed up sampling households from people flow'''
       
        # Preallocate empty array
        household_sizes = np.full(
            flow_people, -1, dtype=np.int16)  # Way too big
        household_types = np.full(
            flow_people, -1, dtype=np.int16)  # Way too big

        i = 0
        
        if household_sizes_dest.size != 0:   
            while flow_people > max_household_size:
                # create random indice to sample from destination population
                indice = np.random.randint(household_sizes_dest.size)
                household_size = household_sizes_dest[indice]
                household_type = household_types_dest[indice]

                # correct for people left in flow (weird things might occur here)
                household_size = min(
                    [flow_people, household_size])

                # subtract from flow
                flow_people -= household_size
                # store
                household_sizes[i] = household_size
                household_types[i] = household_type
                i+=1
            
            # allocate last household and clip array
            household_sizes[i] = flow_people
            household_sizes = household_sizes[:i+1]
            household_types = household_types[:i+1]

        else: # else moving to inland node
            while flow_people > max_household_size:
                    household_size = min(
                        [flow_people, np.random.randint(1, max_household_size)])
                    flow_people -= household_size
                    household_sizes[i] = household_size
                    i += 1
                # allocate last household
            household_sizes[i] = flow_people
            household_sizes = household_sizes[:i+1]        
            household_types = household_types[:i+1]

        return household_sizes

    def ambient_pop_change(self):
        # Process population growth
        population_change = self.agents.population_data.loc[self.geom_id]['change']
        international_migration_pp = self.model.settings['gravity_model']['annual_international_migration'] / np.sum(
            self.agents.regions.population)
        population_change += np.floor(
            international_migration_pp * self.population)

        # No nat pop change in spin up period
        # if self.model.spin_up_flag:
        #     population_change = 0

        # Generate households from new people
        if int(abs(population_change)) > 0:
            household_sizes = self.return_household_sizes(
                int(abs(population_change)), self.max_household_size)
        else: household_sizes = np.array([], np.int16)
        return population_change, household_sizes

    # Generate households moving out of inland nodes
    @staticmethod
    @njit
    def _generate_households(
        n_households_to_move,
        household_sizes,
        move_to_region_per_household,
        init_risk_perception,
        init_risk_aversion
    ):
        # Generate people moving out
        # sum household sizes to find total number of people moving
        n_movers = int(household_sizes.sum())
        # create output arrays people
        to_region = np.full(n_movers, -1, dtype=np.int32)
        household_id = np.full(n_movers, -1, dtype=np.int32)
        gender = np.full(n_movers, -1, dtype=np.int8)
        age = np.full(n_movers, -1, dtype=np.int8)
        income_percentile = np.full(n_households_to_move, -99, dtype=np.int32) # only income percentiles from inland nodes are defined as -99 ### TO DO: Replace with Marijns dataset
        household_type = np.full(n_households_to_move, -99, dtype=np.int32)
        income = np.full(n_movers, 0, dtype=np.float32)
        
        risk_aversion = np.full(
            household_sizes.size,
            init_risk_aversion,
            dtype=np.float32)
        
        risk_perception = np.full(
            household_sizes.size,
            init_risk_perception,
            dtype=np.float32)

        # fill households
        start_idx = 0
        for i in range(n_households_to_move):
            end_idx = start_idx + int(household_sizes[i])
            to_region[start_idx: end_idx] = move_to_region_per_household[i]
            household_id[start_idx: end_idx] = i
            gender[start_idx: end_idx] = np.random.randint(
                0, 2, size=end_idx - start_idx)
            age[start_idx: end_idx] = np.random.randint(
                0, 85, size=end_idx - start_idx)

            start_idx = end_idx

        assert end_idx == n_movers
        return n_movers, to_region, household_id, gender, age, income_percentile, household_type, income, risk_perception, risk_aversion

    def load_arrays_from_npz(self):
        '''To be implemented in child class'''
        pass
    def save_arrays_to_npz(self):
        '''To be implemented in child class'''
        pass