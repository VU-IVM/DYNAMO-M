from agents.nodes import Nodes
from agents.beach_manager import BeachManager
from agents.beaches import Beaches
from agents.government_agent import GovernmentAgent
from agents.insurer_agent import InsurerAgent
from population_change import PopulationChange
from GDP_change import GDP_change
from decision_module import DecisionModule

class Agents:
    '''This class contains all the agents (nodes (households), beaches, and governments) in the model. It is used to initiate the agents and to call the step function of each agent class.'''

    def __init__(self, model):
        self.model = model
        self.agent_types = ['government', 'regions', 'beaches', 'beach_manager']
        self.regions = Nodes(model, self)
        self.beaches = Beaches(model, self)
        self.beach_manager = BeachManager(
            model=model, agents=self, beaches=self.beaches)
        self.population_change = PopulationChange(
            model=self.model, agents=self)
        self.GDP_change = GDP_change(
            model=self.model, agents=self)
        self.decision_module = DecisionModule(model=model, agents=self)
        self.decision_module.load_gravity_models() 
        self.government = GovernmentAgent(
            model=model, agents=self)
        self.insurer = InsurerAgent(model=model,
            agents=self)

    def step(self):
        if self.model.settings['general']['include_ambient_pop_change']:
            self.population_change.step()
        if (not self.model.spin_up_flag or self.model.args.GUI):
            self.GDP_change.step()

        self.beaches.step()
        self.regions.step()
        self.beach_manager.step()
        # self.get_attibute_sizes()

