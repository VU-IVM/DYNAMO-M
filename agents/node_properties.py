import numpy as np

class NodeProperties:

    @property
    def agents_in_simulation(self):
        return [household.n for household in self.all_households]

    @property
    def n_moved_out_last_timestep(self):
        return self._n_moved_out_last_timestep

    @n_moved_out_last_timestep.setter
    def n_moved_out_last_timestep (self, value):
        self._n_moved_out_last_timestep = value

    @property
    def n_moved_in_last_timestep(self):
        return self._n_moved_in_last_timestep

    @n_moved_in_last_timestep.setter
    def n_moved_in_last_timestep(self, value):
        self._n_moved_in_last_timestep = value

    @property
    def ids(self):
        return[household.geom_id for household in self.all_households]

    @property
    def destination_effects_gravity_model(self):
        return [household.destination_effect_gravity_model for household in self.all_households]

    @property
    def centroids(self):
        return np.array([geom['properties']['centroid'] for geom in self.geoms])

    @property
    def perc_people_moved_out(self):
        return [round(household.perc_people_moved_out, 1) for household in self.all_households]

    @property
    def people_moved_out_last_timestep(self):
        return [household.people_moved_out_last_timestep for household in self.all_households]

    @property
    def income_region(self):
        return [household.income_region for household in self.all_households]

    @property
    def average_age_node(self):
        return [household.average_age_node for household in self.all_households]

    @property
    def income_distribution_region(self):
        return [household.income_distribution_region for household in self.all_households]

    @property
    def income_percentiles_regions(self):
        return [household.income_percentile for household in self.all_households]
    
    @property
    def average_household_income(self):
        return [household.average_household_income for household in self.all_households]

    @property
    def population(self):
        return [household.population for household in self.all_households]

    @property
    def coastal_population(self):
        coastal_population = np.sum([household.population if hasattr(household, 'amenity_value') else 0 for household in self.all_households ])
        return coastal_population

    @property
    def summed_beach_amenity(self):
        return [household.summed_beach_amenity for household in self.all_households]

    @property
    def total_shoreline_change_admin(self):
        return [household.total_shoreline_change_admin for household in self.all_households]

    @property
    def amenity_value_regions(self):
        amenity_value_regions = [household.amenity_value if hasattr(household, 'amenity_value') else 0 for household in self.all_households ]
        
        # Fill regions
        return amenity_value_regions

    @property
    def percentage_adapted(self):
        return [household.percentage_adapted for household in self.all_households]

    @property
    def n_households_adapted(self):
        return [household.n_households_adapted for household in self.all_households]

    @property
    def ead_total(self):
        return [household.ead_total for household in self.all_households]

    @property
    def households_near_beach(self):
        return [household.households_near_beach for household in self.all_households]

    @property
    def beach_amenity_dict(self):
        return [household.beach_amenity_dict for household in self.all_households]

    @property
    def segment_IDs_admin(self):
        return [household.segment_IDs_admin for household in self.all_households]

    @property
    def people_near_beach(self):
        return [household.people_near_beach for household in self.all_households]

    @property
    def n_moved_in_last_timestep(self):
        return [household.n_moved_in_last_timestep for household in self.all_households]

    @property
    def n_moved_out_last_timestep(self):
        return [household.n_moved_out_last_timestep for household in self.all_households]

    # @property
    # def agent_locations(self):
    #     return [household.locations for household in self.all_households]

    @property
    def beach_width_admin(self):
        return [household.beach_width_admin for household in self.all_households]
    
    @property
    def nearshore_slopes_admin(self):
        return [household.nearshore_slopes_admin for household in self.all_households]

    @property
    def average_beach_width(self):
        return [household.average_beach_width for household in self.all_households]

    @property
    def beach_length(self):
        return [household.beach_length for household in self.all_households]


    @property
    def household_sizes(self):
        return [household.size for household in self.all_households]

    @property 
    def household_incomes(self):
        return[household.income for household in self.all_households]

    @property 
    def household_adapted(self):
        return[household.adapt for household in self.all_households]

    @property
    def household_risk_perception(self):
        return[household.risk_perception for household in self.all_households]

    @property
    def household_ead(self):
        return[household.ead for household in self.all_households]

    @property
    def since_flood(self):
        return[household.flood_timer for household in self.all_households] 

    @property 
    def flood_tracker(self):
        return[household.flood_tracker for household in self.all_households]  

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

    @property
    def n(self):
        return self.n_households_per_region[self.admin_idx]

    @property
    def n_people(self):
        assert np.count_nonzero(self._people_indices_per_household != -1) == self.size.sum()
        return self.size.sum()

    @property
    def max_n_people(self):
        return self._empty_index_stack.size

    @n.setter
    def n(self, value):
        self.n_households_per_region[self.admin_idx] = value

    @property
    def people_indices_per_household(self):
        return self._people_indices_per_household[:self.n]

    @people_indices_per_household.setter
    def people_indices_per_household(self, value):
        self._people_indices_per_household[:self.n] = value

    @property
    def people_indices(self):
        return self._people_indices_per_household[self._people_indices_per_household != -1]

    @property
    def risk_aversion(self):
        return self._risk_aversion[:self.n]

    @risk_aversion.setter
    def risk_aversion(self, value):
        self._risk_aversion[:self.n] = value

    @property
    def gender(self):
        gender = self._gender[self.people_indices]
        assert (gender != -1).all()
        return gender

    @gender.setter
    def gender(self, value):
        self._gender[self.people_indices] = value

    @property  
    def age(self):
        age = self._age[self.people_indices]
        assert (age != -1).all()
        return age

    @age.setter
    def age(self, value):
        self._age[self.people_indices] = value

    @property
    def amenity_value(self):
        amenity_value = self._amenity_value[:self.n]
        return amenity_value

    @amenity_value.setter
    def amenity_value(self, value):
        self._amenity_value[:self.n] = value
    
    @property
    def beach_proximity_bool(self):
        beach_proximity_bool = self._beach_proximity_bool[:self.n]
        return beach_proximity_bool

    @beach_proximity_bool.setter
    def beach_proximity_bool(self, value):
        self._beach_proximity_bool[:self.n] = value

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
    def income(self,value):
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
    def hh_risk_aversion(self):
        hh_risk_aversion = self._hh_risk_aversion[:self.n]
        return hh_risk_aversion

    @hh_risk_aversion.setter
    def hh_risk_aversion(self, value):
        self._hh_risk_aversion[:self.n] = value

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
    def adapt(self):
        adapt = self._adapt[:self.n]
        return adapt

    @adapt.setter
    def adapt(self, value):
        self._adapt[:self.n] = value

    @property
    def time_adapt(self):
        time_adapt = self._time_adapt[:self.n]
        return time_adapt
    
    @time_adapt.setter
    def time_adapt(self, value):
        self._time_adapt[:self.n] = value

    @property 
    def flood_count(self):
        flood_count = self._flood_count[:self.n]
        return flood_count
    
    @flood_count.setter
    def flood_count(self, value):
        self._flood_count[:self.n] = value

    @property
    def flooded(self):
        flooded = self._flooded[:self.n]
        return flooded
    
    @flooded.setter
    def flooded(self, value):
        self._flooded[:self.n] = value


class InlandNodeProperties:
    @property
    def locations(self):
        return self.geom['properties']['centroid']

    @property
    def n(self):
        return self.n_households_per_region[self.admin_idx]

    @n.setter
    def n(self, value):
        self.n_households_per_region[self.admin_idx] = value