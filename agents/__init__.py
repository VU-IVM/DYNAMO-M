from agents.nodes import Nodes
from agents.government import Government
from agents.beaches import Beaches
from scipy.spatial import distance_matrix as sdistance_matrix
from agents.node_properties import NodeProperties
from population_change import WorldPopProspectsChange
from population_change import SSP_population_change

class Agents:
    def __init__(self, model):
        self.model = model
        self.agent_types = ['regions', 'beaches', 'government']  # not sure where this is used
        self.regions = Nodes(model, self)
        self.beaches = Beaches(model, self)
        self.government = Government(model, self)

    def step(self):     
        self.beaches.step()
        self.regions.step()
        self.government.step()

        if not self.model.spin_up_flag or not self.model.args.headless:
            # Quick fix 
            if self.model.args.ssp == 'coupled':
                if self.model.args.rcp == 'rcp4p5':
                    ssp = 'SSP2'
                elif self.model.args.rcp == 'rcp8p5':
                    ssp = 'SSP5'
                elif self.model.args.rcp == 'control':
                    ssp = 'SSP2'
                else:
                    raise NotImplementedError('Scenario not selected?')
            else:
                ssp = self.model.args.ssp

            if self.model.current_time.year > 2015 and ssp != 'worldpop':
                self.population_data = SSP_population_change(
                    initial_figures=self.model.data.nat_pop_change,
                    SSP_projections = self.model.data.SSP_projections,
                    SSP = ssp,
                    iso3 = self.model.args.area[0], # Not yet working with multiple isocodes. FIX THIS LATER.
                    population=self.regions.population,
                    admin_keys=self.regions.ids,
                    current_year = self.model.current_time.year
                    )
            else:
                self.population_data = WorldPopProspectsChange(
                    initial_figures = self.model.data.nat_pop_change, 
                    HistWorldPopChange = self.model.data.HistWorldPopChange, 
                    WorldPopChange = self.model.data.WorldPopChange, 
                    population = self.regions.population, 
                    admin_keys = self.regions.ids, 
                    year = self.model.current_time.year)