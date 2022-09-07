'''This script loads WorldPop population projections and adjusts simulated annual population growth rate accordingly'''
from scipy.optimize import minimize
import pandas as pd
import numpy as np

def objective(x0, growth_rate, obs_growth_sum, pop):
    adj_growth = growth_rate.copy()
    
    adj_growth[growth_rate > 0] *= 1+x0
    adj_growth[growth_rate < 0] *= 1-x0
    
    adj_growth_sum = round(np.sum(adj_growth * pop))
    
    score = (adj_growth_sum - obs_growth_sum)**2
    return score


def WorldPopProspectsChange(initial_figures, HistWorldPopChange, WorldPopChange, population, admin_keys, year):
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

    

