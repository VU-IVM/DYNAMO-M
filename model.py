from dateutil.relativedelta import relativedelta
from honeybees.library.helpers import timeprint
from honeybees.area import Area
from honeybees.model import Model
from reporter import Reporter
from hazards.flooding.flood_risk import FloodRisk
from agents.coastal_amenities import CoastalAmenities
from data import Data
from agents import Agents
from artists import ArtistsCOASTMOVE
import datetime
import yaml
from random_generator import RandomNumberGenerator
import os
import pickle
import logging

class SLRModel(Model):
    def __init__(
            self,
            config_path,
            settings_path,
            study_area,
            args,
            coordinate_system='WGS84'):

        current_time = datetime.date(2020, 1, 1)
        timestep_length = relativedelta(years=1)
        n_timesteps = None  # Not used
        

        Model.__init__(
            self,
            current_time,
            timestep_length,
            config_path,
            args=args,
            n_timesteps=n_timesteps)
        self.run_id = 0
        self.spin_up_cycle = 0
        self.spin_up_flag = True
        self.calibrate_flag = False
        self.settings_path = settings_path
        with open(self.settings_path) as f:
            self.settings = yaml.load(f, Loader=yaml.FullLoader)

        self.random_module = RandomNumberGenerator(model=self)

        self.current_time = self.config['general']['start_time']
        self.timestep_length = relativedelta(years=1)
        self.end_time = self.config['general']['end_time']

        # Only include spinuptime when run without GUI
        if not self.args.GUI:
            self.spin_up_time = self.config['general']['spin_up_time']
        else:
            self.spin_up_time = 0

        self.timestep = 0
        self.artists = ArtistsCOASTMOVE(self)
        study_area['xmin'] = max(-180, study_area['xmin'])
        study_area['xmax'] = min(180, study_area['xmax'])
        self.area = Area(self, study_area)
        self.data = Data(self)
        self.agents = Agents(self)
        self.reporter = Reporter(self)

        if self.config['general']['create_agents']:
            self.flood_risk = FloodRisk(model=self)
            self.coastal_amenities = CoastalAmenities(model=self)

        # coordinate system must be WGS84. If not, all code needs to be
        # reviewed
        assert coordinate_system == 'WGS84'

        # initiate move logger
        # self._initiate_move_logger()   

        # This variable is required for the batch runner. To stop the model
        # if some condition is met set running to False.
        if self.end_time.year > 2081:
            print("Warning, end time exceeds GLOFRIS inundation projections")
        timeprint("Finished setup")

    def _initiate_move_logger(self):
        self.move_logger = logging.getLogger('move_logger')
        self.move_logger.setLevel(logging.INFO)
        self.move_logger.addHandler(logging.StreamHandler())
        logfile = 'move_logger.log'
        file_handler = logging.FileHandler(logfile, mode='w')
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
        file_handler.setFormatter(formatter)


    def spin_up(self):
        for i in range(self.spin_up_time):
            self.agents.step()
            print(f'spin up {i+1} of {self.spin_up_time}')
            self.reporter.timesteps[i] = f'spin_up_cycle_{i}'
            self.reporter.step()
            self.spin_up_cycle += 1
        # report spinup
        self.spin_up_flag = False
        self.report() # export end of spinup


    def run_model(self):
        # Set flag to false and reset random sets
        self.random_module.reset_all_seeds()

        # print some information
        n_agents_fp = sum([agents.n for agents in self.agents.regions.all_households if agents.geom_id.endswith('flood_plain')])
        print(f'Spin up finished, running with {round(n_agents_fp * 1E-3)} thousand households in floodplains')

        while True:
            print(self.current_time)
            self.step()
            self.timestep += 1

            if self.current_time >= self.end_time:
                break


    def run(self):
        # update export folder if provided in parsed arguments
        if 'report_folder' in self.args:
            self.reporter.export_folder = self.args.report_folder
            self.logger.info(f'report folder set to {self.args.report_folder}')

        if self.spin_up_time > 0 and not self.args.GUI:
            self.spin_up()
        else: 
            self.spin_up_flag = False
        
        self.run_model()

    def spin_up_and_store_model(self, folder_multirun):
        # create folder export spinup
        report_folder_multirun = os.path.join(
            folder_multirun)
        os.makedirs(report_folder_multirun, exist_ok=True)
        
        # set export folder spin up
        self.reporter.export_folder = report_folder_multirun

        # set log file
        self.config['logging']['logfile'] = os.path.join(report_folder_multirun, f'slr_{self.args.rcp}_{self.args.ssp}.log')
        self.logger = self.create_logger()

        if self.spin_up_time > 0 and not self.args.GUI:
            for i in range(self.spin_up_time):
                self.agents.step()
                print(f'spin up {i+1} of {self.spin_up_time}')
                self.reporter.timesteps[i] = f'spin_up_cycle_{i}'
                self.reporter.step()
                self.spin_up_cycle += 1
                self.report() # report each timestep in spin up for debugging.
        else: 
            self.spin_up_flag = False

        # report spinup
        self.report() # export end of spinup
        n_agents_fp = sum([agents.n for agents in self.agents.regions.all_households if agents.geom_id.endswith('flood_plain')])
        print(f'Spin up finished, running with {round(n_agents_fp * 1E-3)} thousand households in floodplains')
        self.spin_up_flag = False

        # also sample water levels for rcp 8.5
        self.args.rcp = 'rcp8p5'
        self.data.load_water_levels()
        [household._initiate_water_level_polynomials_admin() for household in self.agents.regions.all_households if hasattr(household, '_initiate_water_level_polynomials_admin')]
        
        # then delete water levels from model
        [delattr(household, 'water_level_polynomials_admin') for household in self.agents.regions.all_households if hasattr(household, 'water_level_polynomials_admin')]

        # pickle model and save to disk
        if type(self.args.area) == list: area_name = "+".join(self.args.area)
        else: area_name = self.args.area
        path = os.path.join(self.args.low_memory_mode_folder, 'spin_up', f'model_{area_name}_{self.args.admin_level}.pkl')
        if not os.path.exists(path):
            delattr(self, 'data')
            with open(path, 'wb') as f:
                pickle.dump(self, f)
            print('model saved to disk')

    @staticmethod
    def run_from_spinup(args, folder_multirun, run=0, overwrite=False):

        # update report folder for run
        report_folder_multirun = os.path.join(
            folder_multirun,
            'individual_runs',
            f'run_{str(run).zfill(2)}')
        if not os.path.exists(report_folder_multirun):
            os.makedirs(report_folder_multirun, exist_ok=True)

        if not overwrite:
            # check if run has already completed
            if os.path.exists(os.path.join(report_folder_multirun, 'population.csv')):
                print(f'results found in {report_folder_multirun}. Skipping run.')
                return None

        # load model from scratch
        if type(args.area) == list: area_name = "+".join(args.area)
        else: area_name = args.area
        path = os.path.join(args.low_memory_mode_folder, 'spin_up', f'model_{area_name}_{args.admin_level}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                model = pickle.load(f)
                model.args = args
                model.config['logging']['logfile'] = os.path.join(report_folder_multirun, f'slr_{args.rcp}_{args.ssp}.log')
                model.reporter.export_folder = report_folder_multirun
                model.logger = model.create_logger()
                model.data = Data(model)
                # replace erosion rates
                model.agents.beaches._initiate_shoreline_change()  
                # reset GDP and population growth
                model.agents.population_change.__init__(model, model.agents)
                model.agents.GDP_change.__init__(model, model.agents)
                # set model run attribute
                model.run_id = run

            print('model loaded from disk')
        else:
            raise FileNotFoundError('Run spin_up_and_store_model first')
        model.random_module.reset_all_seeds()
        model.spin_up_flag = False

        # update water levels admin cells
        [household._initiate_water_level_polynomials_admin() for household in model.agents.regions.all_households if hasattr(household, '_initiate_water_level_polynomials_admin')]       
        # update scratch to current TMPDIR (since all runs are in the same bash script this should not be needed')
        assert model.args.low_memory_mode_folder == args.low_memory_mode_folder
        model.args.low_memory_mode_folder = args.low_memory_mode_folder
        [household.update_scratch_folder() for household in model.agents.regions.all_households if hasattr(household, 'update_scratch_folder')]
        # update dike heights with correct water levels
        [household._initiate_admin_coastal_dikes() for household in model.agents.regions.all_households if hasattr(household, '_initiate_admin_coastal_dikes')]


        while True:
            print(model.current_time)
            model.step()
            model.timestep += 1
            if model.current_time >= model.end_time:
                model.report()
                break

        return model
    
    @staticmethod
    def load_model_from_disk(path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                model = pickle.load(f)
            model.data = Data(model)
            model.spin_up_flag = False
            print('model loaded from disk')
        else:
            raise FileNotFoundError(
                'Model not found. Rerun or create model using spinup model function'
                )
        return model
