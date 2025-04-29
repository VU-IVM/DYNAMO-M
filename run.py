from model import SLRModel
from honeybees.visualization.canvas import Canvas
from honeybees.visualization.modules import ChartModule
from honeybees.visualization.ModularVisualization import ModularServer
import os
import numpy as np
import faulthandler
from utils.parse_arguments import parse_arguments
from utils.get_study_area import get_study_area
from utils.wrappers import run_with_profiling
faulthandler.enable()
np.seterr(all='raise')


if __name__ == '__main__':
    # parse arguments
    args = parse_arguments()
    
    # get study area (or create if not in cache)
    study_area = get_study_area(area=args.area, subdivision=args.subdivision, admin_level=args.admin_level, coastal_only=args.coastal_only)

    # set some globals
    CONFIG_PATH = args.config
    SETTINGS_PATH = args.settings
    MODEL_NAME = 'SLR'

    # set model params
    model_params = {
        "config_path": CONFIG_PATH,
        "settings_path": SETTINGS_PATH,
        "args": args,
        "study_area": study_area,
    }

    # if not running with GUI...
    if not args.GUI:
        
        # check if model is ran from spinup period
        if args.run_from_cache:
            if type(args.area) == list: area_name = "+".join(args.area)
            else: area_name = args.area
            path = os.path.join(args.low_memory_mode_folder, 'spin_up', f'model_{area_name}_{args.admin_level}.pkl')

            if os.path.exists(path):
                print('Model loaded from cache')
                model = SLRModel.run_from_spinup(args)
                model.report()
            else:
                print(f'Model not found in {path}')
                model = SLRModel(**model_params)
                model.spin_up_and_store_model() 
                del model
                model = SLRModel.run_from_spinup(args)
                model.report()
        
        # if complete model is run check if run with profiling
        else: 
            model = SLRModel(**model_params)
            if args.profiling:            
                run_with_profiling(model)
            else:
                model.run()
                model.report()

    # if run with GUI create GUI and plots
    else:
        series_to_plot = [
            # [
            #     {"name": "ead_nodes",
            #      "ID": f"{admin['properties']['id']}"}
            #     for admin in study_area['admin'] if admin['properties']['id'].endswith('_flood_plain')
            # ],

            # [
            #     {"name": "population",
            #      "ID": f"{admin['properties']['id']}"}
            #     for admin in study_area['admin'] if admin['properties']['id'].endswith('_flood_plain')
            # ],

            [
                {"name": "percentage_adapted",
                 "ID": f"{admin['properties']['id']}"}
                for admin in study_area['admin'] if admin['properties']['id'].endswith('_flood_plain')
            ],

            [
                {"name": "n_households_insured",
                 "ID": f"{admin['properties']['id']}"}
                for admin in study_area['admin'] if admin['properties']['id'].endswith('_flood_plain')
            ],


        ]
        server_elements = [
            Canvas(max_canvas_height=800, max_canvas_width=1200)
        ] + [ChartModule(series) for series in series_to_plot]

        DISPLAY_TIMESTEPS = [
            'year',
            'decade',
            'century'
        ]

        server = ModularServer(
            MODEL_NAME,
            SLRModel,
            server_elements,
            DISPLAY_TIMESTEPS,
            model_params=model_params,
            port=None)
        server.launch(port=args.port, browser=args.browser)
