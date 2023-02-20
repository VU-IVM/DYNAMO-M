'''This script loads WorldPop population projections and adjusts simulated annual population growth rate accordingly'''
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np

def objective(
    x0: float,
    growth_rate: np.ndarray,
    proj_pop: float,
    pop: list
    ) -> float:

    '''This is the objective function used to optimize population growth to match the simulated population growth to the population growth projected in World population prospects
    
    Args:
        x0: initial parameter setting
        growth_rate: initial department level growth rate
        proj_pop: targetted projected population
        pop: population currently residing in the country
    
    Returns:
        score: the score of the objective function
    '''
       
    adj_growth = growth_rate.copy()  
    adj_growth[growth_rate > 0] *= 1+x0
    adj_growth[growth_rate < 0] *= 1-x0
    adj_growth_sum = round(np.sum(adj_growth * pop))
    score = (adj_growth_sum - proj_pop)**2
    return score


def SSP_population_change(
    initial_figures,
    SSP_projections,
    SSP,
    iso3,
    population,
    admin_keys,
    current_year
    ):
    raise NotImplementedError('Only applied in paper 2')
    # Filter dataset on iso3 code and SSP
    data_filt = SSP_projections[SSP_projections['Region'] == iso3]
    data_filt = data_filt[data_filt['Scenario'] == SSP]
    years_in_dataframe = np.array([year for year in data_filt.columns if type(year) == int])
    population_in_dataframe = np.array(data_filt[years_in_dataframe])[0]

    ambient_change = []
    
    if current_year >= np.min(years_in_dataframe):
        interpolater = interp1d(x = years_in_dataframe, y = population_in_dataframe)       
        population_change = (interpolater(current_year+1) - interpolater(current_year)) * 1E6

    else:
        raise NotImplementedError('include historical change')

    for region, pop in zip(admin_keys, population):
        ambient_change.append(round(initial_figures.loc[region]['ambient_change'] * 0.01, 6))



    growth_rate = np.array(ambient_change)
    pop = population
    obs_growth_sum = population_change
    args = (growth_rate, obs_growth_sum, pop)

    x0 = 0.2

    # Optimize adjustion factor               
    res = minimize(objective, x0, method = 'nelder-mead', args = args)
    factor = res.x

    # Adjust growth figures
    adj_growth = growth_rate.copy()
    adj_growth[growth_rate > 0] *= 1 + factor
    adj_growth[growth_rate < 0] *= 1 - factor

    # Calculate pop change
    simulated_population_change_scaled = np.int32(adj_growth * population)
    
    # Export 
    population_change = pd.DataFrame({'change': simulated_population_change_scaled}, index=admin_keys)
    return population_change


def WorldPopProspectsChange(
    initial_figures: pd.core.frame.DataFrame, 
    HistWorldPopChange: pd.core.frame.DataFrame, 
    WorldPopChange: pd.core.frame.DataFrame, 
    population: list, 
    admin_keys: list, 
    year: int
    ) -> pd.core.frame.DataFrame:

    '''This function is used to generate population growth and decline figures based on an optimization of departmental growth rate figures.
    
    Args:
        initial_figures: department level population growth rates.
        HistWorldPopChange: global historical population figures.
        WorldPopChange: global population projections.
        population: list of population residingin in each node (coastal and inland)
        admin_keys: admin names based on gadm dataset.
        year: current year in the simulation timestep.
        
    Returns:
        population_change: pandas DataFrame showing the adjusted absolute natural population change in each node'''


    # Read Insee net population change
    initial_figures = pd.read_csv(r'DataDrive/POPULATION/ambient_population_change_gadm_2.csv', index_col='keys')

    ambient_change = []

    # Select historical observations or future projections
    if year < 2020:
        FRA_PopChange = HistWorldPopChange[HistWorldPopChange[HistWorldPopChange.columns[2]] == 'France']
        FRA_change = (int(FRA_PopChange[str(year+1)]) - int(FRA_PopChange[str(year)])) * 1E3

    elif year >= 2020:
        FRA_PopChange = WorldPopChange[WorldPopChange[WorldPopChange.columns[2]] == 'France']
        FRA_change = (int(FRA_PopChange[str(year+1)]) - int(FRA_PopChange[str(year)])) * 1E3

    for region, pop in zip(admin_keys, population):
        ambient_change.append(round(initial_figures.loc[region]['ambient_change'] * 0.01, 6))


    # Initialize optimization
    
    growth_rate = np.array(ambient_change)
    pop = population
    obs_growth_sum = FRA_change
    args = (growth_rate, obs_growth_sum, pop)

    x0 = 0.2

    # Optimize adjustion factor               
    res = minimize(objective, x0, method = 'nelder-mead', args = args)
    factor = res.x

    # Adjust growth figures
    adj_growth = growth_rate.copy()
    adj_growth[growth_rate > 0] *= 1 + factor
    adj_growth[growth_rate < 0] *= 1 - factor

    # Calculate pop change
    simulated_population_change_scaled = np.int32(adj_growth * population)


    
    # Export 
    population_change = pd.DataFrame()
    population_change['keys'] = admin_keys
    population_change['change'] = simulated_population_change_scaled
    population_change  = population_change.set_index('keys')
    return population_change

    

