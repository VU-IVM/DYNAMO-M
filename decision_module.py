from dataclasses import dataclass
from functools import cache
from joblib import Parallel
from numba.core.decorators import njit
import numpy as np
import yaml


# def EU_stay(decision_params):
#     return EU_do_nothing(**decision_params), EU_adapt(**decision_params)

@njit(cache=True)
def IterateThroughFlood(n_floods, wealth, income, amenity_value, max_T, expected_damages, n_agents, r):
 
    # Allocate array
    NPV_summed = np.full((n_floods+3, n_agents), -1, dtype=np.float32)


    # Iterate through all floods 
    for i, index in enumerate(np.arange(1, n_floods+3)):
         # Check if we are in the last iterations
        if i < n_floods:
            NPV_flood_i =  wealth + income + amenity_value - expected_damages[i]
            NPV_flood_i = (NPV_flood_i).astype(np.float32)

        # if in the last two iterations do not subtract damages (probs of no flood event)
        elif i >= n_floods: 
            NPV_flood_i =  wealth + income + amenity_value
            NPV_flood_i = NPV_flood_i.astype(np.float32)


        # iterate over NPVs for each year in the time horizon and apply time discounting
        NPV_t0 = (wealth + income + amenity_value).astype(np.float32) # no flooding in t=0
        NPV_tx = np.full(NPV_t0.size, -1, dtype=np.float32) # Array

        for t in np.arange(1, max_T):
            NPV_tx += NPV_flood_i/ (1 + r)**t

        # Add NPV at t0 (which is not discounted)
        NPV_tx += NPV_t0

        # Store result
        NPV_summed[index] = NPV_tx

    # Store NPV at p=0 for bounds in integration 
    NPV_summed[0] = NPV_summed[1]   

    return NPV_summed

def calcEU_no_nothing(
    n_agents,
    wealth,
    income,
    amenity_value,
    risk_perception,
    expected_damages,
    adapted,
    p_floods,
    T,
    r,
    sigma,
    **kwargs): 

     
    # Ensure p floods is in increasing order
    indices = np.argsort(p_floods)
    expected_damages = expected_damages[indices]
    p_floods = np.sort(p_floods)

    # Preallocate arrays
    n_floods, n_agents = expected_damages.shape
    p_all_events = np.full((p_floods.size + 3, n_agents), -1, dtype=np.float32) 

    # calculate perceived risk
    perc_risk = p_floods.repeat(n_agents).reshape(p_floods.size, n_agents)
    perc_risk *= risk_perception
    p_all_events[1:-2, :] = perc_risk

    # Cap percieved probability at 0.998. People cannot percieve any flood event to occur more than once per year
    if np.max(p_all_events > 0.998):
        p_all_events[np.where(p_all_events > 0.998)] = 0.998 

    # Add lasts p to complete x axis to 1 for trapezoid function (integrate domain [0,1])
    p_all_events[-2, :] = p_all_events[-3, :] + 0.001
    p_all_events[-1, :] = 1
    
    # Add 0 to ensure we integrate [0, 1]
    p_all_events[0, :] = 0
    
    # Prepare arrays
    max_T = np.int32(np.max(T))

    # Part njit, iterate through floods
    n_agents = np.int32(n_agents)
    NPV_summed = IterateThroughFlood(n_floods, wealth, income, amenity_value, max_T, expected_damages, n_agents, r)

    # Filter out negative NPVs
    NPV_summed = np.maximum(0, NPV_summed)

    if (NPV_summed == 0).any():
        print(f'Warning, {np.sum(NPV_summed == 0)} negative NPVs encountered')

    # Calculate expected utility
    if sigma == 1:
        EU_store = np.log(NPV_summed)
    else:
        EU_store = (NPV_summed ** (1-sigma) )/ (1-sigma)

    # Use composite trapezoidal rule integrate EU over event probability
    y = EU_store
    x = p_all_events
    EU_do_nothing_array = np.trapz(y=y, x=x, axis=0)
    EU_do_nothing_array[np.where(adapted == 1)] = -np.inf # People who already adapted cannot not adapt

    return EU_do_nothing_array

@njit(cache=True)
def calcEU_adapt(
    expendature_cap,
    loan_duration,
    n_agents,
    sigma,
    wealth,
    income,
    amenity_value,
    p_floods,
    risk_perception,
    expected_damages_adapt,
    adaptation_costs,
    time_adapted,
    adapted,
    T,
    r,

    # Not used (kwargs not supported in njit)
    lifespan_dryproof,
    expected_damages
    ):
    
    # Preallocate arrays
    EU_adapt =  np.full(n_agents, -1,  dtype=np.float32)
    EU_adapt_dict =  np.zeros(len(p_floods) + 3, dtype=np.float32)

    t = np.arange(0, np.max(T), dtype=np.int32)

    # preallocate mem for flood array
    p_all_events = np.empty((p_floods.size + 3) , dtype=np.float32)

    # Ensure p floods is in increasing order
    indices = np.argsort(p_floods)
    expected_damages_adapt = expected_damages_adapt[indices]
    p_floods = np.sort(p_floods)


    # Identify agents unable to afford dryproofing
    constrained = np.where(income * expendature_cap <= adaptation_costs)
    unconstrained = np.where(income * expendature_cap > adaptation_costs)
    
    # Those who cannot affort it cannot adapt
    EU_adapt[constrained] = -np.inf

    # Iterate only through agents who can afford to adapt
    for i in unconstrained[0]:

        # Find damages and loan duration left to pay
        expected_damages_adapt_i = expected_damages_adapt[:, i].copy()
        payment_remainder = max(loan_duration - time_adapted[i], 0)

        # Extract decision horizon
        t_agent = t[:T[i]]

        # NPV under no flood event
        NPV_adapt_no_flood = np.full(T[i], wealth[i] + income[i] + amenity_value[i], dtype=np.float32)
        NPV_adapt_no_flood[:payment_remainder] -= adaptation_costs[i]
        NPV_adapt_no_flood = np.sum(NPV_adapt_no_flood / (1+r)**t_agent)
        
        # Apply utility function to NPVs
        if sigma == 1: 
            EU_adapt_no_flood = np.log(NPV_adapt_no_flood)
        else: 
            EU_adapt_no_flood = (NPV_adapt_no_flood ** (1-sigma))/ (1-sigma)
      
        # Calculate NPVs outcomes for each flood event
        NPV_adapt = np.full((p_floods.size, T[i]), wealth[i] + income[i] + amenity_value[i], dtype=np.float32)
        NPV_adapt[:, 1:] -= expected_damages_adapt_i.reshape((p_floods.size, 1))
        NPV_adapt[:, :payment_remainder] -= adaptation_costs[i]
        
        NPV_adapt /= (1 + r) ** t_agent
        NPV_adapt_summed = np.sum(NPV_adapt, axis=1, dtype=np.float32)
        NPV_adapt_summed = np.maximum(np.full(NPV_adapt_summed.shape, 0, dtype=np.float32), NPV_adapt_summed) # Filter out negative NPVs
        
        if (NPV_adapt_summed == 0).any():
            print(f'Warning, {np.sum(NPV_adapt_summed == 0)} negative NPVs encountered')
        
        # Calculate expected utility
        if sigma == 1:
            EU_adapt_flood = np.log(NPV_adapt_summed)
        else:
            EU_adapt_flood = (NPV_adapt_summed ** (1-sigma)) / (1-sigma)
        
        # Store results
        EU_adapt_dict[1:EU_adapt_flood.size+1] = EU_adapt_flood       
        EU_adapt_dict[p_floods.size+1: p_floods.size+3] = EU_adapt_no_flood
        EU_adapt_dict[0] = EU_adapt_flood[0]

        # Adjust for percieved risk 
        p_all_events[1:p_floods.size+1] = risk_perception[i] * p_floods
        p_all_events[-2:]  = p_all_events[-3] + 0.001,  1. 
        
        # Ensure we always integrate domain [0, 1]
        p_all_events[0] = 0

        # Integrate EU over probabilities trapezoidal 
        EU_adapt[i] = np.trapz(EU_adapt_dict, p_all_events)
    
    return EU_adapt

# Cost function
@njit(cache=True)
def LogisticFunction(Cmax, k, x):
    y = Cmax/ (1+np.exp(-k * x))
    return y



# njit not faster here, maybe include when considering more destinations
def fill_regions(regions_select, income_distribution_regions, income_percentile, expected_income_agents):
    for i, region in enumerate(regions_select):
        
        # Sort income
        sorted_income = np.sort(income_distribution_regions[region])
        
        # Derive indices from percentiles
        income_indices = np.floor(income_percentile * 0.01 * len(sorted_income)).astype(np.int32)
        
        # Extract based on indices
        expected_income_agents[i] = sorted_income[income_indices]

@njit(cache=True)
def fill_NPVs(regions_select, wealth, expected_income_agents, amenity_value_regions, n_agents, distance, max_T, r, t_arr, sigma, Cmax, cost_shape):    
    
    # Preallocate arrays
    NPV_summed = np.full((regions_select.size, n_agents), -1, dtype=np.float32)
    
    # Fill NPV arrays for migration to each region
    for i, region in enumerate(regions_select):
        
        # Add wealth and expected incomes. Fill values for each year t in max_T
        NPV_t0 = (wealth + expected_income_agents[i, :] + amenity_value_regions[region]).astype(np.float32)
        
        # Apply and sum time discounting
        NPV_region_discounted = np.zeros(NPV_t0.size, dtype=np.float32)

        # Iterate and sum
        for t in range(max_T):
            t = float(t)
            NPV_region_discounted += NPV_t0/(1+r)**t

        # Subtract migration costs (these are not time discounted, occur at t=0) 
        NPV_region_discounted -= LogisticFunction(Cmax=Cmax, k=cost_shape, x=distance[region])

        # Store time discounted values
        NPV_summed[i, :] =  NPV_region_discounted

    # Calculate expected utility
    if sigma == 1:
        EU_regions = np.log(NPV_summed)
    else:
        EU_regions = NPV_summed ** (1-sigma) / (1-sigma)
    return EU_regions

def EU_migrate(
    regions_select,
    n_agents,
    sigma,
    wealth,
    income_distribution_regions,
    income_percentile, 
    amenity_value_regions,
    distance,
    Cmax,
    cost_shape,
    T,
    r):
    
    '''This function calculates the subjective expected utilty of migration and the region for which 
    utility is highers. The function loops through each agent, processing their individual characteristics.
    
    Args:
        region_id: ID of the current region
        n_agents (int): The number of agents to loop through.
        sigma (float): Array containing the risk aversion of each agent.
        wealth (float): Array containing the wealth of each agent.
        income_region (float): Array containing the mean income in each region.
        income_percentile (int): Array Array containing the income percentile of each agent. This percentile indicates their position in the lognormal income distribution.
        amenity_value_regions (float): Array containing the amenity value of each region. 
        travel_cost (float): Array containing the travel costs to each region. The travel costs are assumed to be te same for each agent residing in the same region. 
        fixed_migration_costs: Array containing the fixed migration costs for each agent. 
        T (int): Array containing the decision horizon of each agent.
        r (float): Array containing the time preference of each agent.
    
    Returns:
        EU_migr_MAX (float): Array containing the maximum expected utility of migration of each agent.
        ID_migr_MAX (float): Array containing the region IDs for which the utility of migration is highest for each agent. 

    '''
    # Preallocate arrays
    # EU_migr = np.full(len(regions_select), -1, dtype=np.float32)
    max_T = np.int32(np.max(T))
    EU_migr_MAX = np.full(n_agents, -1, dtype=np.float32)
    ID_migr_MAX = np.full(n_agents, 1, dtype=np.float32)
    # NPV_discounted = np.full(regions_select.size, -1, np.float32)

    # Preallocate decision horizons
    t_arr = np.arange(0, np.max(T), dtype=np.int32)
    regions_select=  np.sort(regions_select)

    # Fill array with expected income based on position in income distribution    
    expected_income_agents = np.full((regions_select.size, income_percentile.size), -1, dtype=np.float32)

    fill_regions(regions_select, income_distribution_regions, income_percentile, expected_income_agents)    
    EU_regions = fill_NPVs(regions_select, wealth, expected_income_agents, amenity_value_regions, n_agents, distance, max_T, r, t_arr, sigma, Cmax, cost_shape)
    
    EU_migr_MAX = EU_regions.max(axis=0)
    region_indices = EU_regions.argmax(axis=0)

    # Indices are relative to selected regions, so extract
    ID_migr_MAX = np.take(regions_select, region_indices)
    
    return EU_migr_MAX, ID_migr_MAX

admin_level = 2
# Load yml with parameter settings
with open(f'gravity_module_gadm_{admin_level}.yml') as f:
    gravity_settings = yaml.load(f, Loader=yaml.FullLoader)

model = 'gravity_model_pop_inc_dist_coast2'


constant = gravity_settings[model]['intercept']
B_pop_i = gravity_settings[model]['population_i']
B_pop_j = gravity_settings[model]['population_j']
B_inc_i = gravity_settings[model]['income_i']
B_inc_j = gravity_settings[model]['income_j']
B_distance = gravity_settings[model]['distance']
B_coastal_i = gravity_settings[model]['coastal_i']
B_coastal_j = gravity_settings[model]['coastal_j']



def gravity_model(pop_i=None, pop_j=None,
                     inc_i=None, inc_j=None,
                     coastal_i=None, coastal_j=None, 
                     distance=None,
                     B_pop_i=B_pop_i,
                     B_pop_j=B_pop_j, B_inc_i=B_inc_i,
                     B_inc_j=B_inc_j, B_coastal_i=B_coastal_i, B_coastal_j=B_coastal_j,
                     B_distance=B_distance, constant=constant):			

    flow = np.exp(
        B_pop_i * np.log(pop_i) + 
        B_pop_j * np.log(pop_j) + 
        B_inc_i * np.log(inc_i) +
        B_inc_j * np.log(inc_j) +       
        B_coastal_i * coastal_i + 
        B_coastal_j * coastal_j  +
        B_distance * np.log(distance) +
        constant) 
    # flow = 0
    return round(flow)
