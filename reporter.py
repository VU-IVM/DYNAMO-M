from honeybees.reporter import Reporter
from typing import DefaultDict, Union, Any
import numpy as np

class Reporter(Reporter):
    def __init__(self, model, subfolder: Union[None, str] = None) -> None:
        super().__init__(model, subfolder)

    def extract_agent_data(self, name: str, conf: dict) -> None:
        """This method is used to extract agent data and apply the relevant function to the given data.
        
        Args:
            name: Name of the data to report.
            conf: Dictionary with report configuration for values.
        """
        agents = getattr(self.model.agents, conf['type'])
        try:
            values = getattr(agents, conf['varname']) # Included a defaulting value of nan
        except AttributeError:
            raise AttributeError(f"Trying to export '{conf['varname']}', but no such attribute exists for agent type '{conf['type']}'")
        if 'split' in conf and conf['split']:
            for ID, admin_values in zip(agents.ids, values):
                self.parse_agent_data((name, ID), admin_values, agents, conf)
        else:
            self.parse_agent_data(name, values, agents, conf)
