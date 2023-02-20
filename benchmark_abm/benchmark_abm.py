''' Calibration  of the COASTMOVE ABM.
We benchmark the risk perception of households to match observed adaptation uptake in France.
We calibrate the gravity model of migration between inland nodes and towards coastal nodes on observed migration flows'''


from operator import mul
import sys
import os

# Add parent folder to directory
sys.path.insert(1, os.path.abspath('.'))

import numpy as np
import pandas as pd
import yaml
from honeybees.argparse import parser
from run import get_study_area    
from model import SLRModel
import datetime


def InitiateCalibration():
    # Initiate parsed arguments (defaults)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--area', dest='area', type=str)
    group.add_argument('--iso3', dest='area', nargs="+", type=str)
    parser.add_argument('--profiling', dest='profiling', default=False, action='store_true')
    parser.add_argument('--admin_level', dest='admin_level', type=int, default=2)
    parser.add_argument('--ssp', dest='ssp', type=str, default='worldpop', help='Choose SSP1-5 or worldpop')
    parser.add_argument('--iterations', dest='iterations', type=int, default=1, help="choose the number of model iterations")
    parser.add_argument('--rcp', dest='rcp', type=str, default='control', help=f"choose between control, rcp4p5 and rcp8p5")
    parser.add_argument('--coastal-only', dest='coastal_only', action='store_true', help=f"only run the coastal areas")
    parser.add_argument('--config', dest='config', type=str, default='config.yml', help='Choose to config file for the model run.')
    parser.add_argument('--settings', dest='settings', type=str, default='settings.yml', help='Choose to settings file for the model run.')
    parser.add_argument('--split_id', dest='split_id', type=int, default=0, help='subdivide the model runs for simple batch parallellization.')

    parser.set_defaults(headless=True)

    args = parser.parse_args()
    # args.coastal_only = True
    study_area = get_study_area(args.area, args.admin_level, args.coastal_only)

    # Load parameter ranges
    ParamRangesPath = os.path.join('benchmark_abm', 'parameter_ranges.csv')
    ParamRanges = pd.read_csv(ParamRangesPath, sep=",").set_index('Parameter')
    ParamRanges = ParamRanges[ParamRanges['Use'] == True].drop('Use', axis=1)

    params = list(ParamRanges.index)
    parameter_space = {}
    for param in params:
        parameter_space[param] = np.round(np.arange(ParamRanges.loc[param]['MinValue'], ParamRanges.loc[param]['MaxValue'], ParamRanges.loc[param]['Step'], np.float32), 3)
    
    a  = 1
    for key in parameter_space.keys():
        a *= parameter_space[key].size
    print(f'n_runs: {a}')

    if args.split_id == 1:  
        parameter_space['max_risk_perception'] = np.array([1, 2])

    if args.split_id == 2:
        parameter_space['max_risk_perception'] = np.array([3, 4])

    if args.split_id == 3:
        parameter_space['max_risk_perception'] = np.array([5, 6])
    

    return args, study_area, parameter_space

def IndividualRun(model, calibration_results_runs, duplicate):
    model.calibrate_flag = False # set to false to include migration in spinup period
    model.run()

    # Filter Nones to only get coastal nodes
    adapted_in_region = [perc for perc in model.agents.regions.percentage_adapted if perc is not None]
    adapted_agents = [x for x in model.agents.regions.household_adapted if x is not None]
    adapted_all_regions = np.array([], dtype = np.float32)
    adapted_agents_all = np.array([], dtype = np.int8)


    # Extract all regions from list and combine in single np array
    for adapted_reg, adapted_agent in zip(adapted_in_region, adapted_agents):
        adapted_all_regions = np.append(adapted_all_regions, adapted_reg)
        adapted_agents_all = np.append(adapted_agents_all, adapted_agent)
    
    perc_adapted = np.mean(adapted_agents_all, dtype=np.float32) * 100
    calibration_results_runs[duplicate] = perc_adapted

def RunModel(args, study_area, parameter_space):
    # Loop through each parameter combination? For now test with 100 combinations
    # Set globals
    run_folder = os.path.join('DataDrive', 'BENCHMARK', f'runs_{args.split_id}')
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
    

    with open(os.path.join('settings.yml')) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join('config.yml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    settings_template = settings.copy()
    config_template = config.copy()
    calibration_results_dict = {}
    calibration_results_df = {}
    calibration_results_df[args.split_id] = {}
    run = 0

    for max_risk_perception in parameter_space['max_risk_perception']:
        for expenditure_cap in parameter_space['expenditure_cap']:
            for loan_duration in parameter_space['loan_duration']:
                for interest_rate in parameter_space['interest_rate']:
                    # Initiate settings
                    
                    settings['general']['flood']['spin_up_flood'] = False
                    settings['general']['flood']['random_flood'] = False
                    settings['general']['flood']['year'] = 2016


                    settings['general']['include_migration'] = False
                    settings['general']['include_adaptation'] = True
                    settings['general']['dynamic_behavior'] = True

                    # Only include few regions in migration to speed up things
                    settings['decisions']['regions_included_in_migr'] = 10
                    
                    # Initiate geopandas dataframe
                    # Do a grid search

                    settings_template['flood_risk_calculations']['risk_perception']['max'] = float(max_risk_perception)
                    if max_risk_perception == 0:
                        settings_template['flood_risk_calculations']['risk_perception']['min'] = float(0)
                    
                    settings_template['adaptation']['interest_rate'] = round(float(interest_rate), 3)
                    settings_template['adaptation']['loan_duration'] = round(float(loan_duration), 3)
                    settings_template['adaptation']['expenditure_cap'] = round(float(expenditure_cap), 3)


                    SETTINGS_PATH = os.path.join(run_folder, 'settings.yml')
                    CONFIG_PATH = os.path.join(run_folder, 'config.yml')
                    
                    # Dump template to yaml
                    with open(SETTINGS_PATH, 'w') as outfile:
                        yaml.dump(settings_template, outfile, default_flow_style=False)


                    # Change config
                    config_template['general']['start_time'] = datetime.datetime(2015, 1, 1)
                    config_template['general']['end_time'] = datetime.datetime(2018, 1, 1)
                    if args.coastal_only:
                        config_template['general']['spin_up_time'] = 0
                    else:
                        config_template['general']['spin_up_time'] = 10
                    config_template['general']['size'] = 'xxlarge'
                    config_template['general']['report_folder'] = os.path.join(run_folder, 'report')

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

                    # Run 5 times (duplicates)
                    duplicates = 1
                    calibration_results_runs = np.full(duplicates, -1, dtype = np.float32)
                    duplicate_ids = np.arange(0, duplicates) # Iterable 
                    for i in duplicate_ids:
                        model = SLRModel(**model_params)
                        IndividualRun(model=model, calibration_results_runs=calibration_results_runs, duplicate=i)
                   
                    calibration_results_perc = np.mean(calibration_results_runs[calibration_results_runs != -1], dtype=np.float32)
                    
                    if len(calibration_results_dict) == 0:  
                        calibration_results_dict['loan_duration'] = loan_duration
                        calibration_results_dict['expenditure_cap'] = expenditure_cap
                        calibration_results_dict['max_risk'] = max_risk_perception
                        calibration_results_dict['interest_rate'] = interest_rate
                        calibration_results_dict['perc_adapted'] = calibration_results_perc
                        calibration_results_df[args.split_id]  = pd.DataFrame(calibration_results_dict, index=[0])
                        run += 1

                    else:
                        run_result = {}
                        run_result['loan_duration'] = loan_duration
                        run_result['expenditure_cap'] = expenditure_cap
                        run_result['max_risk'] = max_risk_perception
                        run_result['interest_rate'] = interest_rate
                        run_result['perc_adapted'] = calibration_results_perc
                        run_result_df = pd.DataFrame(run_result, index=[run])
                        calibration_results_df[args.split_id]  = pd.concat([calibration_results_df[args.split_id] , run_result_df])
                        run += 1
                
                        calibration_results_df[args.split_id]['run'] = calibration_results_df[args.split_id].index
                        calibration_results_df[args.split_id]  = calibration_results_df[args.split_id] .set_index('run')
                        
                        if args.split_id != 0:
                            calibration_results_df[args.split_id].to_csv(os.path.join(run_folder, f'benchmark_output_part_{args.split_id}.csv')) # Export and overwrite each of max risk to be safe
                        else: 
                            calibration_results_df[args.split_id].to_csv(os.path.join(run_folder, 'benchmark_output.csv')) # Export and overwrite each of max risk to be safe
                        print('Runs saved')

if __name__ == '__main__':
    args, study_area, parameter_space = InitiateCalibration()
    RunModel(args, study_area, parameter_space)