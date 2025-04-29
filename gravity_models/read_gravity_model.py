import yaml
import os
from pathlib import Path


def read_gravity_model(
    region: int,
) -> dict:
    '''This function loads the gravity models yml as dictionaries

    Args:
        admin_level: admin level of the model run

    Returns:
        gravity_settins: a dictionary containing the coeefficients of the estimated gravity model
    '''
    # Get parent directory of current file to load (required for docs)
    parent = Path(__file__).parent
    fn_model = os.path.join(parent, 'regions', f'{region}.yml')
    if os.path.exists(fn_model):
        with open(fn_model) as f:
            gravity_settings = yaml.load(f, Loader=yaml.FullLoader)
    else:
        print(f'no gravity model found for {region}, loading default')
        fn_model = os.path.join(parent, 'countries', f'default_regional.yml')
        with open(fn_model) as f:
            gravity_settings = yaml.load(f, Loader=yaml.FullLoader)

    return gravity_settings
