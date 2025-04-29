from honeybees.reporter import Reporter
from typing import DefaultDict, Union, Any
import numpy as np
import pandas as pd
from collections.abc import Iterable
import os 
from copy import deepcopy

class Reporter(Reporter):
    def __init__(self, model, subfolder: Union[None, str] = None) -> None:
        super().__init__(model, subfolder)

    def report(self) -> dict:
        """This method can be called to save the data that is currently saved in memory to disk."""
        report_dict = {}
        # check if report folder exists, else create
        if not os.path.exists(self.export_folder):
            os.makedirs(self.export_folder)
        for name, values in self.variables.items():
            if isinstance(values, dict):
                df = pd.DataFrame.from_dict(values)
                df.index = self.timesteps
            elif isinstance(values[0], Iterable):
                df = pd.DataFrame.from_dict(
                    {
                        k: v
                        for k, v in zip(self.timesteps, values)
                    }
                )
            else:
                df = pd.DataFrame(values, index=self.timesteps, columns=[name])
            if 'format' not in self.model.config['report'][name]:
                raise ValueError(f"Key 'format' not specified in config file for {name}")
            export_format = self.model.config['report'][name]['format']
            filepath = os.path.join(self.export_folder, name + '.' + export_format)
            if export_format == 'csv':
                df.dropna(how='all', axis=1).to_csv(filepath) # drop no data from export dataframe
            elif export_format == 'xlsx':
                df.to_excel(filepath)
            elif export_format == 'npy':
                np.save(filepath, df.values)
            elif export_format == 'npz':
                np.savez_compressed(filepath, data=df.values)
            else:
                raise ValueError(f'save_to format {export_format} unknown')
        return report_dict


    def parse_agent_data(self, name: str, values: Any, agents, conf: dict) -> None:
        """This method is used to apply the relevant function to the given data.
        
        Args:
            name: Name of the data to report.
            values: Numpy array of values.
            agents: The relevant agent class.
            conf: Dictionary with report configuration for values.
        """
        function = conf['function']
        if function is None:
            values = deepcopy(values)  # need to copy item, because values are passed without applying any a function.
            self.report_value(name, values, conf)
        else:
            function, *args = conf['function'].split(',')
            if function == 'mean':
                value = np.mean(values)
                self.report_value(name, value, conf)
            elif function == 'sum':
                value = np.sum([value for value in values if value != None]) # account for None values in inland nodes
                self.report_value(name, value, conf)
            elif function == 'sample':
                sample = getattr(agents, "sample")
                value = values[sample]
                for s, v in zip(sample, value):
                    self.report_value((name, s), v, conf)
            elif function == 'groupcount':
                for group in args:
                    group = int(group)
                    self.report_value((name, group), np.count_nonzero(values == group), conf)
            else:
                raise ValueError(f'{function} function unknown')