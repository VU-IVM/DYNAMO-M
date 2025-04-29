from honeybees.agents import AgentBaseClass
import numpy as np

class InsurerAgent(AgentBaseClass):
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

    def derive_premium(self, coastal_node):
        # simply take the average EAD per household in node
        if hasattr(coastal_node, 'ead_total'):
            coastal_node.insurance_premium = coastal_node.ead_total/ coastal_node.n
    
    def step(self, coastal_node):
        self.derive_premium(coastal_node)