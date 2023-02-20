import scipy
import numpy as np

def calculate_beach_amenity(
    agent_wealth,
    coastal_amenity_function,
    beach_width: np.ndarray,
    beach_proximity_bool: bool,
    )-> np.ndarray:

    '''Function used to calculate the amenity value of a beach. 
    Args:
        coastal_amenity_function (scipy.interpolate.interpolate.interp1d): function used to calculate the amenity value of a beach
        beach_width (np.ndarray): beach width closest to the agent.
        beach_proximity_bool (bool): boolean indicating whether the agent is located adjecent to a beach or not.
        max_beach_amenity_value (int): maximum amenity value of a beach.
    Returns:
        beach_amenity: amenity value of a beach per agent in euros.

    '''
    

    # just a quick linear function to interpolate amenity value based on shoreline retreat         
    beach_amenity_function = scipy.interpolate.interp1d(x = coastal_amenity_function.index, y = coastal_amenity_function['premium'])
    
    # Filter values outside interpolation range
    beach_width_adj = np.maximum(beach_width, min(coastal_amenity_function.index))
    beach_width_adj = np.minimum(beach_width_adj, max(coastal_amenity_function.index))
    
    # Set values to zero (default)
    beach_amenity = np.zeros(beach_width_adj.size, dtype=np.float32)

    # Calculate
    calculated_beach_amenity = beach_amenity_function(beach_width_adj[beach_proximity_bool]) * agent_wealth[beach_proximity_bool]
    beach_amenity[beach_proximity_bool] = calculated_beach_amenity
    


    # Iterate over individual beaches and determine associated amenity stream

    return beach_amenity


def calculate_coastal_amenity(
    coastal_amenity_function, 
    distance_to_coast: np.ndarray,
    agent_wealth: np.ndarray
    )-> np.ndarray:
    
    '''This function calculates the coastal amenity value experienced by the agent based on distance to coast, erosion rate and wealth.
    
    Args:
        distance_to_coast (np.ndarray): distance to coast in meters
        shoreline_change (np.ndarray): shoreline change in meters per year
        agent_wealth (np.ndarray): agent wealth in euros
    
    Returns:
        np.ndarray: amenity value in euros
    '''
    # Initiate amenity distance decay function (interpolate between values of Conroy & Milosch 2011)
    
    coastal_function = scipy.interpolate.interp1d(x = coastal_amenity_function.index, y = coastal_amenity_function['premium'])
    
    # Filter values outside interpolation range
    distance_to_coast = np.maximum(distance_to_coast, min(coastal_amenity_function.index))
    distance_to_coast = np.minimum(distance_to_coast, max(coastal_amenity_function.index))      
    
    return coastal_function(distance_to_coast) * agent_wealth

def total_amenity(
        coastal_amenity_functions, 
        beach_proximity_bool, 
        dist_to_coast_raster,
        agent_locations, 
        beach_width,
        agent_wealth,
        ):
        


    distance_to_coast = dist_to_coast_raster.sample_coords(agent_locations)

    coastal_amenity = calculate_coastal_amenity(
        coastal_amenity_function = coastal_amenity_functions['dist2coast'],
        distance_to_coast = distance_to_coast,
        agent_wealth = agent_wealth
        )

    if beach_proximity_bool.sum() > 0:
        beach_amenity =  calculate_beach_amenity(
            agent_wealth = agent_wealth,
            coastal_amenity_function = coastal_amenity_functions['beach_amenity'],
            beach_width=beach_width,
            beach_proximity_bool=beach_proximity_bool,

            )

    else:
        beach_amenity = 0

    # return amenity value
    amenity_value = coastal_amenity + beach_amenity
    summed_beach_amenity = np.sum(beach_amenity)
    return amenity_value, beach_amenity
    