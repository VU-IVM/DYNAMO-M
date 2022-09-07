from dateutil.relativedelta import relativedelta
from honeybees.library.helpers import timeprint
from time import time
from honeybees.area import Area
from honeybees.model import Model
from honeybees.reporter import Reporter
from data import Data
from agents import Agents
from artists import Artists, ArtistsCOASTMOVE
import datetime
from dateutil.relativedelta import relativedelta
import sys
from export_agents import export_pop_change


class SLRModel(Model):
    def __init__(self, config_path, settings_path, study_area, args, coordinate_system='WGS84'):

        current_time = datetime.date(2020, 1, 1)
        timestep_length = relativedelta(years=1)
        n_timesteps = None # Not used

        Model.__init__(self, current_time, timestep_length, config_path, args=args, n_timesteps=n_timesteps)
        self.spin_up_cycle = 0   
        self.spin_up_flag = True  
        self.calibrate_flag = False  
        self.settings_path = settings_path
        self.current_time = self.config['general']['start_time']
        self.timestep_length = relativedelta(years=1)
        self.end_time = self.config['general']['end_time']
        self.spin_up_time = self.config['general']['spin_up_time']
        self.timestep = 0
        self.artists = ArtistsCOASTMOVE(self)
        self.area = Area(self, study_area)
        self.data = Data(self)
        self.agents = Agents(self)
        self.reporter = Reporter(self)

        
        assert coordinate_system == 'WGS84'  # coordinate system must be WGS84. If not, all code needs to be reviewed

        # This variable is required for the batch runner. To stop the model
        # if some condition is met set running to False.
        if self.end_time.year > 2081:
            print("Warning, end time exceeds GLOFRIS inundation projections")
        timeprint("Finished setup")

    def run(self):
        for i in range(self.spin_up_time):
            self.agents.step()
            print(f'spin up {i+1} of {self.spin_up_time}')
            self.spin_up_cycle += 1
        
        # Set flag to false
        self.spin_up_flag = False
        
        while True:
            print(self.current_time)
            self.step()
            self.timestep += 1
            if self.current_time >= self.end_time:
                # export_pop_change(self.config['general']['start_time'].year, self.end_time.year-1) # TESTER
                break

