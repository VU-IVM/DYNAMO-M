# In this module stochastic flood events are simulated.
import numpy as np 
import rasterio
import rasterio.mask
import fiona
import os
from scipy import interpolate
import pandas as pd

def stochastic_flood(water_levels, return_periods,
 flooded, flood_count, risk_perceptions,
  flood_timer, risk_perc_min, risk_perc_max,
   risk_decr, settings, current_year, spin_up_flag, flood_tracker):
    '''This function simulates stochastic flooding using a random number generator. In its current 
    implementation only one flood event can occur per year.
     
    Args:
        water_levels: A dictionary containing arrays of inundation levels for each agent associated with various return periods.
        return_periods: An array containing the return periods of flood events included in this model.
        flooded: An array containing the flood status of each agent, with 0 indicating not flooded, and 1 indicating flooded.
        flood_count: An array keeping track of the flood experience of each agent.
        risk_perceptions: An array containing the flood risk perceptions of each agent
        method: random if random draw, or a single year.
    Returns:
        flooded: The updated array containing the flood status of each agent
        flood_count: The updated array containing the flood experiance of each agent
        risk_perceptions: The updated array containing the risk perception of each agent.'''

    # update flood timer for all households
    flood_timer += 1
    
    if settings['random_flood']:
        # No flooding in spin up
        if not settings['spin_up_flood'] and spin_up_flag:
            pass
        
        else:
        # Simulate flooding based on random draw
            random_draw = np.random.rand()
            for rt in return_periods:
                if random_draw < (1/rt):
                    flooded[water_levels[rt]>0] = 1
                    flood_count[water_levels[rt]>3] += 1 # Count times people have experienced waterlevels of more than 3 meters (Not used)
                    flood_timer[water_levels[rt]>0] = 0 # Set flood timer to zero for those who experienced flooding
                    flood_tracker = rt
                    break 
    else:
        year = settings['year']
        rt = settings['rt']
        
        if current_year == year:
            flooded[water_levels[rt]>0] = 1
            flood_count[water_levels[rt]>3] += 1 # Count times people have experienced waterlevels of more than 3 meters (Not used)
            flood_timer[water_levels[rt]>0] = 0 # Set flood timer to zero for those who experienced flooding


    risk_perceptions =  risk_perc_max * 1.6 ** (risk_decr * flood_timer) + risk_perc_min
    return flooded, flood_count, risk_perceptions, flood_timer, flood_tracker