import yaml
import os 
from pathlib import Path

def read_gravity_model(
    admin_level: int
    ) -> dict:
    '''This function loads the gravity models yml as dictionaries
    
    Args:
        admin_level: admin level of the model run
    
    Returns:
        gravity_settins: a dictionary containing the coeefficients of the estimated gravity model
    '''
    # Get parent directory of current file to load (required for docs)
    parent = Path(__file__).parent
    with open(os.path.join(parent, f'gravity_module_gadm_{admin_level}.yml')) as f:
        gravity_settings = yaml.load(f, Loader=yaml.FullLoader)
    return gravity_settings
