import pandas as pd
import numpy as np
import pickle
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import yaml
# from linearmodels import PanelOLS
import statsmodels.formula.api as smf
# try mars
import geopandas as gpd

def load_gravity_dataset(path = os.path.join('DataDrive', 'GRAVITY', 'INSEE', 'PROCESSED')):
        with open(os.path.join(path, 'variables_FRA.pickle'), 'rb') as pickle_file:
            explanatory_variables_dict = pickle.load(pickle_file)
        
        gravity_dataset = pd.DataFrame(explanatory_variables_dict)
        flows = pd.read_csv(os.path.join(path, 'department_migration_flows_2017.csv'))        
        gravity_dataset['origin'] = flows['origin']
        gravity_dataset['destination'] = flows['destination']
        gravity_dataset['flow']  = flows['flow']
        return pd.DataFrame(gravity_dataset)

def export_location_effects(params):
    # load gadm for translation
    gadm = gpd.read_file(os.path.join('DataDrive', 'SLR', 'GADM', 'GADM_2.shp'))
    location_effects = {}
    location_effects['origin_effect'] = {}
    location_effects['destination_effect'] = {}

    for param in params.keys():
        if param.startswith('origin'):
            dep_name = param[9:-1]
            dep_gadm = gadm.set_index('NAME_2').loc[dep_name]['GID_2']
            location_effects['origin_effect'][dep_name] = params[param]
        elif param.startswith('destination'):
            dep_name = param[14:-1]
            dep_gadm = gadm.set_index('NAME_2').loc[dep_name]['GID_2']
            location_effects['destination_effect'][dep_name] = params[param]
    
    # make pandas df and export
    location_effects_df = pd.DataFrame(location_effects)
    location_effects_df.to_csv(os.path.join('DataDrive', 'GRAVITY', 'location_effects_OLS.csv'), index=True)       



def OLS_regression(gravity_dataset, omit_zero_flow = True):
    if omit_zero_flow: gravity_dataset = gravity_dataset[gravity_dataset['flow'] > 0]
    x = gravity_dataset[['population_i', 'population_j', 'income_i', 'income_j', 'coastal_i', 'coastal_j', 'distance', 'flow']]
    # x = x.sort_values('flow', ascending=True)

    # log transformation for all columns except coastal dummy
    for column in x.columns:
        if not column.startswith('coast'):
            x[column] = np.maximum(np.log(x[column]), 0)

    # add origin and destination fixed effects
    x['origin'] = pd.Categorical(gravity_dataset['origin'])
    x['destination'] = pd.Categorical(gravity_dataset['destination'])

    # specify model
    model = smf.ols(formula = 'flow ~ population_i + population_j + income_i + income_j + coastal_i + coastal_j + distance', data = x).fit()

    # print results
    print(model.summary())

    # store summary
    with open('summary_fixed_effects.txt', 'w') as fh:
        fh.write(model.summary().as_text())

    # store model
    fn = os.path.join('DataDrive', 'GRAVITY', 'gravity_model_location_effects.yml')
    f = open(fn, 'w+')
    estimated_params = {}
    estimated_params['gravity_model_OLS'] = {}
    for factor in ['age_i', 'age_j', 'population_i', 'population_j', 'income_i', 'income_j', 'coastal_i', 'coastal_j', 'distance']:
        if factor in model.params.keys():
            estimated_params['gravity_model_OLS'][factor] = float(model.params[factor])
        else: estimated_params['gravity_model_OLS'][factor] = .0
    estimated_params['gravity_model_OLS']['intercept'] = float(model.params.Intercept)
    yaml.dump(estimated_params, f, allow_unicode=True)
    f.close()

    # export location effects


    params = model.params
    export_location_effects(params)
    estimated_flow = np.exp(model.predict(x))

    return estimated_flow, params, model

def prepare_gravity_model(params, gravity_dataset):
    coefs = np.zeros(len(params.keys()), np.float32)
    coefs[0] = params.Intercept

    # filter out fixed effects
    param_list = [param for param in params.keys()[1:] if not (param.startswith('origin') or param.startswith('destination'))]
    
    # initiate array explanatory variables 
    explanatory_variables = np.zeros((params.size-1,gravity_dataset.shape[0]), np.float32)

    # log transformation and fill array
    for i, key in enumerate(param_list):
        coefs[i+1] = params[key]
        if not key.startswith('coast'):
            explanatory_variables[i, :] = np.log(gravity_dataset[key])
        else: explanatory_variables[i, :] = gravity_dataset[key]
    
    return coefs, explanatory_variables

def gravity_model(coefs, explanatory_variables):
    alpha = coefs[0]
    betas = coefs[1:].reshape(coefs.size-1, 1)
    flow = np.exp(np.sum(explanatory_variables*betas, axis=0)+alpha)
    return flow

if __name__ == '__main__':
    gravity_dataset = load_gravity_dataset()
    estimated_flow, params, model = OLS_regression(gravity_dataset)
