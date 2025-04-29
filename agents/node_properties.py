import numpy as np


class NodeProperties:  
    
    @property
    def n_household_agents(self):
        return [household.n for household in self.all_households]
   
    @property
    def property_value_nodes(self):
        return [household.property_value_node if hasattr(household, 'property_value_node') else 0 for household in self.all_households]

    @property
    def n_available_housing(self):
        return [household.n_available_housing if household.geom_id.endswith('flood_plain') else household.max_pop - household.population for household in self.all_households]

    @property
    def admin_full(self):
        return [household.admin_full for household in self.all_households]

    @property
    def percentage_moved(self):
        return [household.percentage_moved for household in self.all_households]

    @property
    def median_inc_percentile_region(self):
        return [np.median(household.income_percentile) if hasattr(household, 'income_percentile') else None for household in self.all_households]
    
    @property 
    def population_in_100yr_floodplain(self):
        return [household.population_in_100yr_floodplain if hasattr(household, 'population_in_100yr_floodplain') else None for household in self.all_households]

    @property
    def sd_income_region(self):
        return [np.std(household.income) if hasattr(household, 'income') else None for household in self.all_households]

    @property
    def annual_summed_adaptation_costs(self):
        return [household.annual_summed_adaptation_costs if hasattr(household, 'annual_summed_adaptation_costs') else None for household in self.all_households]

    @property
    def CBA_ratio_upgrade(self):
        return [household.CBA_ratio_upgrade if hasattr(household, 'CBA_ratio_upgrade') else None for household in self.all_households]
   
    @property
    def CBA_ratio_maintain(self):
        return [household.CBA_ratio_maintain if hasattr(household, 'CBA_ratio_maintain') else None for household in self.all_households]


    @property
    def total_population_in_model(self):
        return np.sum([household.population for household in self.all_households])

    @property
    def agents_in_simulation(self):
        return [household.n for household in self.all_households]

    @property
    def fraction_intending_to_migrate(self):
        return [household.fraction_intending_to_migrate if hasattr(household, 'fraction_intending_to_migrate') else None for household in self.all_households]  

    @property
    def n_intending_to_migrate(self):
        return [household.n_intending_to_migrate if hasattr(household, 'n_intending_to_migrate') else None for household in self.all_households]  

    @property
    def fraction_intending_to_migrate_in_model(self):
        n_households_intending_to_migrate = np.sum(
            [household.n_intending_to_migrate if hasattr(household, 'n_intending_to_migrate') else 0 for household in self.all_households]) 
        n_households_in_coastal_nodes = np.sum([household.household_ids_initial_100yr_floodplain.size if hasattr(household, 'household_ids_initial_100yr_floodplain') else 0 for household in self.all_households])
        if n_households_in_coastal_nodes > 0:
            return n_households_intending_to_migrate / n_households_in_coastal_nodes
        else:
            return 0
    @property
    def n_persons_moving_out(self):
        return [household.n_persons_moving_out if hasattr(household, 'n_persons_moving_out') else None for household in self.all_households]

    @property
    def n_persons_moving_in(self):
        return [household.n_persons_moving_in if hasattr(household, 'n_persons_moving_in') else None for household in self.all_households]

    @property
    def n_households_that_would_have_adapted(self):
        return [household.n_households_that_would_have_adapted if hasattr(household, 'n_households_that_would_have_adapted') else None for household in self.all_households]

    @property
    def n_people_that_would_have_adapted(self):
        return [household.n_people_that_would_have_adapted if hasattr(household, 'n_people_that_would_have_adapted') else None for household in self.all_households]

    @property
    def n_households_that_would_have_moved(self):
        return [household.n_households_that_would_have_moved if hasattr(household, 'n_households_that_would_have_moved') else None for household in self.all_households]

    @property
    def n_people_that_would_have_moved(self):
        return [household.n_people_that_would_have_moved if hasattr(household, 'n_people_that_would_have_moved') else None for household in self.all_households]

    @property
    def n_adapted_movers(self):
        return [household.n_adapted_movers if hasattr(household, 'n_adapted_movers') else None for household in self.all_households]

    @property
    def fraction_adapted_movers(self):
        return [household.fraction_adapted_movers if hasattr(household, 'fraction_adapted_movers') else None for household in self.all_households]

    @property
    def n_moved_out_last_timestep(self):
        return self._n_moved_out_last_timestep

    @n_moved_out_last_timestep.setter
    def n_moved_out_last_timestep(self, value):
        self._n_moved_out_last_timestep = value

    @property
    def n_moved_in_last_timestep(self):
        return self._n_moved_in_last_timestep

    @n_moved_in_last_timestep.setter
    def n_moved_in_last_timestep(self, value):
        self._n_moved_in_last_timestep = value

    @property
    def ids(self):
        return [household.geom_id for household in self.all_households]

    @property
    def centroids(self):
        return np.array([geom['properties']['centroid']
                        for geom in self.geoms])

    @property
    def perc_people_moved_out(self):
        return [round(household.perc_people_moved_out, 1)
                for household in self.all_households]

    @property
    def people_moved_out_last_timestep(self):
        return [
            household.people_moved_out_last_timestep for household in self.all_households]

    @property
    def income_region(self):
        return [household.income_region for household in self.all_households]

    @property
    def average_income_node(self):
        return [np.round(np.mean(household.income)) if hasattr(household, 'income') else np.round(np.mean(household.income_distribution_region)) for household in self.all_households]

    @property
    def median_income_node(self):
        return [np.round(np.median(household.income)) if hasattr(household, 'income') else np.round(np.median(household.income_distribution_region)) for household in self.all_households]


    @property
    def average_age_node(self):
        return [household.average_age_node for household in self.all_households]

    @property
    def income_distribution_region(self):
        return [
            household.income_distribution_region for household in self.all_households]

    @property
    def income_percentiles_regions(self):
        return [household.income_percentile if hasattr(household, 'income_percentile') else np.array(
            [-9999]) for household in self.all_households]

    @property
    def average_household_income(self):
        return [
            household.average_household_income for household in self.all_households]

    @property
    def population(self):
        return [household.population for household in self.all_households]

    @property
    def population_in_flood_plain(self):
        population_in_flood_plain = np.sum([
            household.population if household.geom_id.endswith('flood_plain') else 0 for household in self.all_households
            ])
        return population_in_flood_plain

    @property
    def summed_beach_amenity(self):
        return np.sum([household.beach_amenity for household in self.all_households if hasattr(
            household, 'beach_amenity')])

    @property
    def total_shoreline_change_admin(self):
        return [
            household.total_shoreline_change_admin for household in self.all_households]

    @property
    def percentage_adapted(self):
        return [household.percentage_adapted for household in self.all_households]

    @property
    def percentage_insured(self):
        return [household.percentage_insured for household in self.all_households]

    @property
    def coastal_flood_protection(self):
        return [household.coastal_fps for household in self.all_households]

    @property
    def n_households_insured(self):
        return [household.n_households_insured if hasattr(household, 'n_households_insured') else None for household in self.all_households]

    @property
    def n_households_adapted(self):
        return [household.n_households_adapted for household in self.all_households]

    @property
    def fraction_adapted_model(self):
        n_adapted_total = np.sum([household.n_households_adapted for household in self.all_households if household.n_households_adapted != None])
        n_total = np.sum([household.n_households_exposed for household in self.all_households if household.geom_id.endswith('flood_plain')])
        if n_total > 0:
            return n_adapted_total / n_total
        
    @property
    def ead_nodes(self):
        return [household.ead_total for household in self.all_households]

    @property
    def ead_residential_nodes(self):
        return [household.ead_residential for household in self.all_households]

    @property
    def ead_residential_land_use_nodes(self):
        return [household.ead_residential_land_use if hasattr(household, 'ead_residential_land_use') else None for household in self.all_households ]
    
    @property
    def ead_commercial_land_use_nodes(self):
        return [household.ead_commercial_land_use if hasattr(household, 'ead_commercial_land_use') else None for household in self.all_households]

    @property
    def ead_industrial_land_use_nodes(self):
        return [household.ead_industrial_land_use if hasattr(household, 'ead_industrial_land_use') else None for household in self.all_households]

    @property
    def household_adaptation_spendings(self):
        return [household.household_spendings if hasattr(household, 'household_spendings') else None for household in self.all_households]
    
    @property
    def household_adaptation_spendings_relative_to_gdp(self):
        return [household.household_spendings_relative_to_gdp if hasattr(household, 'household_spendings_relative_to_gdp') else None for household in self.all_households]

    @property
    def households_near_beach(self):
        return [
            household.households_near_beach if hasattr(
                household,
                'households_near_beach') else None for household in self.all_households]

    @property
    def segment_IDs_admin(self):
        return [household.segment_IDs_admin for household in self.all_households]

    @property
    def population_near_beach(self):
        return [
            household.people_near_beach if hasattr(
                household, 'people_near_beach') else None for household in self.all_households]

    @property
    def total_population_near_beach(self):
        array = np.array(self.population_near_beach) # lead to double execution of ead_near_beach property. Could be more efficient. 
        return np.sum(array[array!=None])

    @property
    def ead_households_near_beach(self):
        return [household.ead_households_near_beach if hasattr(household, 'ead_households_near_beach') else None for household in self.all_households  ]
  

    @property 
    def total_ead_households_near_beach(self):
        array = np.array(self.ead_households_near_beach) # lead to double execution of ead_near_beach property. Could be more efficient. 
        return np.sum(array[array!=None])

    @property
    def n_adapted_near_beach(self):
        n_adapted_near_beach = []
        for household in self.all_households:
            if hasattr(household, 'beach_proximity_bool'):
                beach_mask = household.beach_proximity_bool == 1
                adapted = household.adapt[beach_mask]
                # assert (adapted != -1).all()
                adapted = adapted[adapted != -1]
                if adapted.size > 0: n_adapted_near_beach.append(adapted.sum())
                else: n_adapted_near_beach.append(0)
            else:
                n_adapted_near_beach.append(None)
        return n_adapted_near_beach

    @property
    def total_n_adapted_near_beach(self):
        return np.sum([n_adapted_near_beach for n_adapted_near_beach in self.n_adapted_near_beach if n_adapted_near_beach != None])

    @property
    def percentage_adapted_near_beach(self):
        percentage_adapted = []
        for household in self.all_households:
            if hasattr(household, 'beach_proximity_bool'):
                beach_mask = household.beach_proximity_bool == 1
                adapted = household.adapt[beach_mask]
                # assert (adapted != -1).all()
                adapted = adapted[adapted != -1]
                if adapted.size > 0: percentage_adapted.append(adapted.mean() * 100)
                else: percentage_adapted.append(0)
            else:
                percentage_adapted.append(None)
        return percentage_adapted

    @property
    def total_percentage_adapted_near_beach(self):
        
        all_household_adapted = np.concatenate(
            [household.adapt for household in self.all_households if hasattr(household, 'adapt')])
        all_beach_masks = np.concatenate(
            [household.beach_proximity_bool == 1 for household in self.all_households if hasattr(household, 'adapt')])
        all_household_adapted = all_household_adapted[all_household_adapted != -1]
        all_beach_masks = all_beach_masks[all_household_adapted != -1]

        if all_household_adapted.size > 0:
            return all_household_adapted[all_beach_masks].mean() * 100
        else:
            return 0
     
    @property
    def n_moved_in_last_timestep(self):
        return [
            household.n_moved_in_last_timestep for household in self.all_households]

    @property
    def n_moved_out_last_timestep(self):
        return [
            household.n_moved_out_last_timestep for household in self.all_households]

    @property
    def all_household_sizes(self):
        return [household.size if hasattr(household, 'size') else None for household in self.all_households ]

    @property
    def all_household_types(self):
        return [household.hh_type if hasattr(household, 'hh_type') else None for household in self.all_households]

    @property
    def household_incomes(self):
        return [household.income for household in self.all_households if household.geom_id.endswith('flood_plain')]

    @property
    def total_percentage_adapted(self):
        all_household_adapted = [household.adapt for household in self.all_households if hasattr(household, 'adapt')]
        if len(all_household_adapted) > 0: 
            all_household_adapted = np.concatenate(all_household_adapted)
            all_household_adapted = all_household_adapted[all_household_adapted != -1]
            return all_household_adapted.mean() * 100
        else:
            return 0

    @property
    def household_risk_perception(self):
        return [household.risk_perception for household in self.all_households if household.geom_id.endswith('flood_plain')]

    @property
    def household_ead(self):
        return [household.ead for household in self.all_households if hasattr(household, 'ead')]

    @property
    def household_adapted(self):
        return [household.adapt for household in self.all_households if hasattr(household, 'adapt')]

    @property
    def since_flood(self):
        return [household.flood_timer for household in self.all_households if hasattr(household, 'flood_timer')]

    @property
    def flood_tracker(self):
        return [
            household.flood_tracker if hasattr(
                household,
                'flood_timer') else None for household in self.all_households]

    @property
    def all_coastal_fps_gov(self):
        list_coastal_fps_dicts = [household.coastal_fps_gov for household in self.all_households if hasattr(household, 'coastal_fps_gov')]
        # merge dictionaries
        return {k: v for d in list_coastal_fps_dicts for k, v in d.items()}


class CoastalNodeProperties:
    @property
    def ids(self):
        if self.n > np.iinfo(np.uint32).max:
            dtype = np.uint64
        else:
            dtype = np.uint32
        return np.arange(0, self.n, dtype=dtype)

    @property
    def activation_order(self):
        return np.arange(self.n)

    @property
    def locations(self):
        return self._locations[:self.n]

    @locations.setter
    def locations(self, value):
        self._locations[:self.n] = value

    @property
    def size(self):
        return self._size[:self.n]

    @size.setter
    def size(self, value):
        self._size[:self.n] = value

    @property
    def hh_type(self):
        return self._hh_type[:self.n]

    @hh_type.setter
    def hh_type(self, value):
        self._hh_type[:self.n] = value

    @property
    def ead(self):
        return self._ead[:self.n]

    @ead.setter
    def ead(self, value):
        self._ead[:self.n] = value

    @property
    def ead_dryproof(self):
        return self._ead_dryproof[:self.n]

    @ead_dryproof.setter
    def ead_dryproof(self, value):
        self._ead_dryproof[:self.n] = value

    @property
    def household_id_per_person(self):
        return self._household_id_per_person[self.people_indices]

    @household_id_per_person.setter
    def household_id_per_person(self, value):
        self._household_id_per_person[self.people_indices] = value
        if isinstance(self._household_id_per_person, np.memmap):
            self._household_id_per_person.flush()

    @property
    def n(self):
        return self.n_households_per_region[self.admin_idx]

    @property
    def n_people(self):
        assert np.count_nonzero(
            self._people_indices_per_household != -
            1) == self.size.sum()
        return self.size.sum()

    @property
    def max_n_people(self):
        return self._empty_index_stack.size

    @n.setter
    def n(self, value):
        if value < 0:
            self.model.report()
            raise ValueError(f'n households becomes negative (was {self.n}, becomes {value}) in {self.geom_id}')
        self.n_households_per_region[self.admin_idx] = value

    @property
    def people_indices_per_household(self):
        return self._people_indices_per_household[:self.n]

    @people_indices_per_household.setter
    def people_indices_per_household(self, value):
        self._people_indices_per_household[:self.n] = value
        if isinstance(self._people_indices_per_household, np.memmap):
            self._people_indices_per_household.flush()

    @property
    def people_indices(self):
        return self._people_indices_per_household[self._people_indices_per_household != -1]

    @property
    def gender(self):
        gender = self._person_attribute_array[0, self.people_indices]
        assert (gender != -1).all()
        return gender

    @gender.setter
    def gender(self, value):
        self._person_attribute_array[0, self.people_indices] = value

    @property
    def age(self):
        age = self._person_attribute_array[1, self.people_indices]
        assert (age != -1).all()
        return age

    @age.setter
    def age(self, value):
        self._person_attribute_array[1, self.people_indices] = value

    @property
    def amenity_value(self):
        amenity_value = self._amenity_value[:self.n]
        return amenity_value

    @amenity_value.setter
    def amenity_value(self, value):
        self._amenity_value[:self.n] = value

    @property
    def distance_to_coast(self):
        distance_to_coast = self._distance_to_coast[:self.n]
        return distance_to_coast

    @distance_to_coast.setter
    def distance_to_coast(self, value):
        self._distance_to_coast[:self.n] = value

    @property
    def beach_proximity_bool(self):
        beach_proximity_bool = self._beach_proximity_bool[:self.n]
        return beach_proximity_bool

    @beach_proximity_bool.setter
    def beach_proximity_bool(self, value):
        self._beach_proximity_bool[:self.n] = value

    @property
    def beach_amenity(self):
        beach_amenity = self._beach_amenity[: self.n]
        return beach_amenity

    @beach_amenity.setter
    def beach_amenity(self, value):
        self._beach_amenity[: self.n] = value

    @property
    def wealth(self):
        wealth = self._wealth[:self.n]
        return wealth

    @wealth.setter
    def wealth(self, value):
        self._wealth[:self.n] = value

    @property
    def income(self):
        income = self._income[:self.n]
        return income

    @income.setter
    def income(self, value):
        self._income[:self.n] = value

    @property
    def income_percentile(self):
        income_percentile = self._income_percentile[:self.n]
        return income_percentile

    @income_percentile.setter
    def income_percentile(self, value):
        self._income_percentile[:self.n] = value

    @property
    def property_value(self):
        property_value = self._property_value[:self.n]
        return property_value

    @property_value.setter
    def property_value(self, value):
        self._property_value[:self.n] = value

    @property
    def decision_horizon(self):
        decision_horizon = self._decision_horizon[:self.n]
        return decision_horizon

    @decision_horizon.setter
    def decision_horizon(self, value):
        self._decision_horizon[:self.n] = value

    @property
    def risk_aversion(self):
        risk_aversion = self._risk_aversion[:self.n]
        return risk_aversion

    @risk_aversion.setter
    def risk_aversion(self, value):
        self._risk_aversion[:self.n] = value

    @property
    def risk_perception(self):
        risk_perception = self._risk_perception[:self.n]
        return risk_perception

    @risk_perception.setter
    def risk_perception(self, value):
        self._risk_perception[:self.n] = value

    @property
    def flood_timer(self):
        flood_timer = self._flood_timer[:self.n]
        return flood_timer

    @flood_timer.setter
    def flood_timer(self, value):
        self._flood_timer[:self.n] = value

    @property
    def adaptation_costs(self):
        adaptation_costs = self._adaptation_costs[:self.n]
        return adaptation_costs

    @adaptation_costs.setter
    def adaptation_costs(self, value):
        self._adaptation_costs[:self.n] = value

    @property
    def adapt(self):
        adapt = self._adapt[:self.n]
        return adapt

    @adapt.setter
    def adapt(self, value):
        self._adapt[:self.n] = value

    @property 
    def would_have_moved(self):
        would_have_moved = self._would_have_moved[:self.n]
        return would_have_moved

    @would_have_moved.setter
    def would_have_moved(self, value):
        self._would_have_moved[:self.n] = value

    @property
    def time_adapt(self):
        time_adapt = self._time_adapt[:self.n]
        return time_adapt

    @time_adapt.setter
    def time_adapt(self, value):
        self._time_adapt[:self.n] = value

    @property
    def flooded(self):
        flooded = self._flooded[:self.n]
        return flooded

    @flooded.setter
    def flooded(self, value):
        self._flooded[:self.n] = value

    @property
    def indice_cell_agent(self):
        indice_cell_agent = self._indice_cell_agent[:self.n]
        return indice_cell_agent

    @indice_cell_agent.setter
    def indice_cell_agent(self, value):
        self._indice_cell_agent[:self.n] = value
    
    @property
    def population_in_country(self):
        populations = np.array(self.agents.regions.population_snapshot)
        regions_in_country = self.select_regions_where_to_move()
        population_in_country = np.sum(populations[regions_in_country]) + populations[self.admin_idx]
        return population_in_country


class InlandNodeProperties:
    @property
    def locations(self):
        return self.geom['properties']['centroid']

    @property
    def n(self):
        return self.n_households_per_region[self.admin_idx]

    @n.setter
    def n(self, value):
        if value < 0:
            self.model.report()
            raise ValueError(f'n households becomes negative (was {self.n}, becomes {value}) in {self.geom_id}')
        self.n_households_per_region[self.admin_idx] = value
    
    @property
    def population_in_country(self):
        populations = np.array(self.agents.regions.population_snapshot)
        regions_in_country = self.select_regions_where_to_move()
        population_in_country = np.sum(populations[regions_in_country]) + populations[self.admin_idx]
        return population_in_country
