'''This script is applied to run the model'''
import os
import numpy as np
import pandas as pd
import rasterio
import pickle

import faulthandler
faulthandler.enable()
# ###################

from honeybees.visualization.ModularVisualization import ModularServer
from honeybees.visualization.modules import ChartModule
from honeybees.visualization.canvas import Canvas
from honeybees.argparse import parser
    
from model import SLRModel

def get_study_area(area, admin_level, coastal_only=False):
    if isinstance(area, list):
        area_name = "+".join(area)
    else:
        area_name = area
    path = f'cache/study_area_{area_name}_{admin_level}{"_coastal_only" if coastal_only else ""}.pickle'

    if not os.path.exists(path):
        import geopandas as gpd
        from shapely.geometry import mapping
        try:
            os.makedirs('cache')
        except OSError:
            pass

        # Load administrative shapefiles
        admin_path = f'DataDrive/SLR/admin/can_flood_gadm_{admin_level}_merged.shp'
        if not os.path.exists(admin_path):
            raise FileNotFoundError("Run find_agent_locations.py first")
        admins = gpd.GeoDataFrame.from_file(admin_path)
        
        # Load GADM areas
        with rasterio.open(f'DataDrive/SLR/admin/can_flood_gadm_{admin_level}.tif', 'r') as src:
            gadm = src.read(1)
            gt = src.transform.to_gdal()

        # Create empty study area dictionary
        study_area = {"name": area_name}

        if isinstance(area, str):
            # Select ISO3 codes based on study area name and set bounding box based on area
            if area == 'benelux':
                iso3 = ["BEL", "NLD", "LUX"]
            elif area == 'se-asia':
                iso3 = ["BGD", 'THA', 'MMR']
            elif area == 'global':
                iso3 = None
            else:
                raise ValueError(f"{area} not recognized")
        elif isinstance(area, list):
            iso3 = area
        else:
            raise ValueError("area must be either list or string")

        if iso3:
            admins = admins[admins['keys'].apply(lambda x: x[:3] in iso3)]

        xmin, ymin, xmax, ymax = 180, 90, -180, -90
        admin_list = []
        # neighbors = {}
        n = len(admins)
        for i, (_, admin) in enumerate(admins.iterrows(), start=1):
            print(f"{i}/{n}")

            gxmin, gymin, gxmax, gymax = admin.geometry.bounds
            xmin = min(xmin, gxmin)
            ymin = min(ymin, gymin)
            xmax = max(xmax, gxmax)
            ymax = max(ymax, gymax)

            centroid = admin.geometry.centroid
            feature = {
                'geometry': admin.geometry.__geo_interface__,
                # 'geometry': vw.simplify_geometry(admin.geometry.__geo_interface__, ratio=0.03),
                'properties': {
                    'id': admin['keys'],
                    'gadm': {
                        'indices': np.where(gadm == admin['ID']),
                        'gt': gt
                    },
                    'centroid': (centroid.x, centroid.y),
                }
            }
            admin_list.append(feature)
            # neighbors = admin[admin.geometry.touches(country.geometry)].index.tolist()
            # feature['properties']['neighbors'] = neighbors


        study_area["admin"] = admin_list
        
        study_area["xmin"] = xmin - gt[1]
        study_area["ymin"] = ymin + gt[5]
        study_area["xmax"] = xmax + gt[1]
        study_area["ymax"] = ymax - gt[5]

        with open(path, 'wb') as f:
            pickle.dump(study_area, f)
    else:
        print('loading study area from cache')
        with open(path, 'rb') as f:
            study_area = pickle.load(f)
    return study_area

if __name__ == '__main__':
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--area', dest='area', type=str)
    group.add_argument('--iso3', dest='area', nargs="+", type=str)
    parser.add_argument('--profiling', dest='profiling', default=False, action='store_true')
    parser.add_argument('--admin_level', dest='admin_level', type=int, default=1)
    parser.add_argument('--iterations', dest='iterations', type=int, default=1, help="choose the number of model iterations")
    parser.add_argument('--rcp', dest='rcp', type=str, default='rcp8p5', help=f"choose between control, rcp4p5 and rcp8p5")
    parser.add_argument('--coastal-only', dest='coastal_only', action='store_true', help=f"only run the coastal areas")
    parser.set_defaults(headless=False)

    args = parser.parse_args()

    study_area = get_study_area(args.area, args.admin_level, args.coastal_only)

    CONFIG_PATH = 'config.yml'
    SETTINGS_PATH = 'settings.yml'

    MODEL_NAME = 'SLR'

    model_params = {
        "config_path": CONFIG_PATH,
        "settings_path": SETTINGS_PATH,
        "args": args,
        "study_area": study_area,
    }
    if args.headless:
        if args.profiling == True:
            print("Run with profiling")
            model = SLRModel(**model_params)
            import cProfile
            import pstats
            with cProfile.Profile() as pr:
                model.run()
            
            with open('profiling_stats.cprof', 'w') as stream:
                stats = pstats.Stats(pr, stream=stream)
                stats.strip_dirs()
                stats.sort_stats('cumtime')
                stats.dump_stats('.prof_stats')
                stats.print_stats()
            pr.dump_stats('profile.prof')    
        

        else:
            model = SLRModel(**model_params)
            model.run()
            report = model.report()

    else:
        series_to_plot = [
            [
                {"name": "ead_total",
                "ID": f"{admin['properties']['id']}"}
                for admin in study_area['admin'] if admin['properties']['id'].endswith('_flood_plain')
            ],

            [
                {"name": "n_moved_out_last_timestep",
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

        server = ModularServer(MODEL_NAME, SLRModel, server_elements, DISPLAY_TIMESTEPS, model_params=model_params, port=None)
        server.launch(port=args.port, browser=args.browser)