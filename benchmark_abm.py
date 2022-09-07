''' Calibration  of the COASTMOVE ABM.
We calibrate the risk perception of households on observed adaptation uptake in France. This script performs a grid search within the parameter space identified in literature.
The dataframes produced here are used in further analysis.
'''

import os
import numpy as np
import pandas as pd
import yaml
from honeybees.argparse import parser
from run import get_study_area    
from model import SLRModel
import datetime
# Set globals
ROOT = r'DataDrive/SLR/calibration'

with open(os.path.join('settings.yml')) as f:
    settings = yaml.load(f, Loader=yaml.FullLoader)

with open(os.path.join('config.yml')) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Load parameter ranges
ParamRangesPath = os.path.join(ROOT, settings['calibrate']['parameter_ranges'])
ParamRanges = pd.read_csv(ParamRangesPath, sep=",", index_col=0)
ParamRanges = ParamRanges[ParamRanges['Use'] == True].drop('Use', axis=1)

# load observed adaptation uptake
ObservedPath = os.path.join(ROOT, settings['calibrate']['observed'])
Observed = pd.read_csv(ObservedPath, sep=",", index_col=0)
Observed = Observed[Observed['Use'] == True].drop('Use', axis=1)

# Initiate parsed arguments (defaults)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--area', dest='area', type=str)
group.add_argument('--iso3', dest='area', type=str)
parser.add_argument('--profiling', dest='profiling', default=False, action='store_true')
parser.add_argument('--admin_level', dest='admin_level', type=int, default=2)
parser.add_argument('--iterations', dest='iterations', type=int, default=1, help="choose the number of model iterations")
parser.add_argument('--rcp', dest='rcp', type=str, default='control', help=f"choose between control, rcp4p5 and rcp8p5")
parser.add_argument('--coastal-only', dest='coastal_only', action='store_true', help=f"only run the coastal areas")
parser.set_defaults(headless=True)

args = parser.parse_args()
study_area = get_study_area(args.area, args.admin_level, args.coastal_only)



def RunModel(targets):
    # Loop through each parameter combination? For now test with 100 combinations
    # Set globals
    ROOT = r'DataDrive/SLR/calibration'
    settings_template = settings.copy()
    config_template = config.copy()



    # Grid search in parameter space
    risk_aversion_array = np.array([1], dtype=np.float32)
    max_risk_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    interest_rate_array = np.arange(0.02, 0.055, 0.005, dtype=np.float32) 
    expenditure_cap_array = np.array([0.025, 0.03, 0.035, 0.04, 0.045, 0.05], dtype= np.float32)
    loan_duration_array = np.arange(10, 32, 2, dtype=np.int32)

    # Initiate geopandas dataframe
    run = 0
    # Do a grid search
    calibration_results_dict = False

    for loan_duration in loan_duration_array: 

        for risk_aversion in risk_aversion_array:

            for expenditure_cap in expenditure_cap_array:

                for max_risk in max_risk_array:

                    for interest_rate in interest_rate_array:

                        # Create folder to store run results and settings file
                        run_id = 'input_1'
                        f_run = os.path.join(ROOT, run_id)
                        if not os.path.exists(f_run):
                            os.mkdir(f_run)
                        
                        # Save model settings
                        # Xynthia in 2010 was associated with a 1 in 100 year water levels
                        settings_template['general']['export_matrix'] = False
                        settings_template['general']['export_move_dictionary'] = False
                        settings_template['general']['include_migration'] = False
                        settings_template['general']['export_agents'] = False 
                        settings_template['general']['flood']['random_flood'] = False
                        settings_template['general']['flood']['year'] = 2017
                        settings_template['general']['flood']['rt'] = 100
                        settings_template['decisions']['risk_aversion'] = int(risk_aversion)
                        settings_template['flood_risk_calculations']['risk_perception']['max'] = float(max_risk)
                        if max_risk == 0:
                            settings_template['flood_risk_calculations']['risk_perception']['min'] = float(0)
                        
                        settings_template['adaptation']['interest_rate'] = float(interest_rate)
                        settings_template['adaptation']['loan_duration'] = float(loan_duration)
                        settings_template['adaptation']['expenditure_cap'] = float(expenditure_cap)


                        SETTINGS_PATH = os.path.join(f_run, 'settings.yml')
                        CONFIG_PATH = os.path.join(f_run, 'config.yml')
                        
                        # Dump template to yaml
                        with open(SETTINGS_PATH, 'w') as outfile:
                            yaml.dump(settings_template, outfile, default_flow_style=False)


                        # Change config
                        config_template['general']['start_time'] = datetime.datetime(2015, 5, 1, 0, 0)
                        config_template['general']['end_time'] = datetime.datetime(2018, 5, 1, 0, 0)
                        config_template['general']['spin_up_time'] = 2
                        config_template['general']['size'] = 'xxlarge'
                        config_template['general']['report_folder'] = os.path.join(f_run, 'report')

                        # Dump template to yaml
                        with open(CONFIG_PATH, 'w') as outfile:
                            yaml.dump(config_template, outfile, default_flow_style=False)

                        # Load and run model
                        model_params = {
                            "config_path": CONFIG_PATH,
                            "settings_path": SETTINGS_PATH,
                            "args": args,
                            "study_area": study_area,
                        }

                        # Run 10 times (duplicates)
                        duplicates = 5
                        calibration_results_rsqrt_runs = np.full(duplicates, -1, dtype = np.float32)
                        calibration_results_runs = np.full(duplicates, -1, dtype = np.float32)

                        for duplicate in range(duplicates):                   
                            
                            model = SLRModel(**model_params)
                            model.calibrate_flag = True
                            model.run()
                        
                            # Extract target
                            target  = targets.loc['AdaptUptake'][1]

                            ## Calculate 'score' TESTTEST
                            # Filter Nones to only get coastal nodes
                            adapted_in_region = [perc for perc in model.agents.regions.percentage_adapted if perc is not None]
                            adapted_agents = [x for x in model.agents.regions.household_adapted if x is not None]
                            adapted_all_regions = np.array([], dtype = np.float32)
                            adapted_agents_all = np.array([], dtype = np.int8)


                            # Extract all regions from list and combine in single np array
                            for adapted_reg, adapted_agent in zip(adapted_in_region, adapted_agents):
                                adapted_all_regions = np.append(adapted_all_regions, adapted_reg)
                                adapted_agents_all = np.append(adapted_agents_all, adapted_agent)
                            
                            score_0 = np.sum((adapted_all_regions - target) **2, dtype=np.float32)  # Squared residuals 
                            score_1 = np.mean(adapted_agents_all, dtype=np.float32) * 100 # Store percentage of all agents having implemented dry flood proofing.

                            calibration_results_rsqrt_runs[duplicate] = score_0
                            calibration_results_runs[duplicate] = score_1


                        calibration_results_rsqrt = np.mean(calibration_results_rsqrt_runs[calibration_results_rsqrt_runs != -1], dtype=np.float32)
                        calibration_results_perc = np.mean(calibration_results_runs[calibration_results_runs != -1], dtype=np.float32)

                        if not calibration_results_dict:  
                            calibration_results_dict = {}
                            calibration_results_dict['loan_duration'] = loan_duration
                            calibration_results_dict['risk_aversion'] = risk_aversion
                            calibration_results_dict['expenditure_cap'] = expenditure_cap
                            calibration_results_dict['max_risk'] = max_risk
                            calibration_results_dict['interest_rate'] = interest_rate
                            calibration_results_dict['perc_adapted'] = calibration_results_perc
                            calibration_results_dict['res_squared'] =  calibration_results_rsqrt
                            calibration_results_df = pd.DataFrame(calibration_results_dict, index=[0])
                            run += 1

                        else:
                            run_result = {}
                            run_result['loan_duration'] = loan_duration
                            run_result['risk_aversion'] = risk_aversion
                            run_result['expenditure_cap'] = expenditure_cap
                            run_result['max_risk'] = max_risk
                            run_result['interest_rate'] = interest_rate
                            run_result['perc_adapted'] = calibration_results_perc
                            run_result['res_squared'] =  calibration_results_rsqrt
                            run_result_df = pd.DataFrame(run_result, index=[run])
                            calibration_results_df = pd.concat([calibration_results_df, run_result_df])
                            run += 1
                    
                            calibration_results_save = calibration_results_df
                            calibration_results_save['run'] = calibration_results_save.index
                            calibration_results_save = calibration_results_save.set_index('run')
                            calibration_results_save.to_csv(os.path.join(ROOT, 'cal_short_results_part_1.csv')) # Export and overwrite each of max risk to be safe
                            print('Runs saved')

if __name__ == '__main__':
    targets = Observed
    RunModel(targets)