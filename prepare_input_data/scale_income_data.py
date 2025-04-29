import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def merge_OECD_WorldBankData():
    fp_oecd = os.path.join('DataDrive', 'ECONOMY', 'OECD_2015_income.csv')
    fp_worldbank = os.path.join('DataDrive', 'ECONOMY', 'adjusted_net_national_income_pc_WoldBank.csv')

    # read
    oecd_income = pd.read_csv(fp_oecd, index_col='Country Name')
    world_bank_income = pd.read_csv(fp_worldbank, index_col='Country Name').dropna()
    world_bank_income = world_bank_income[world_bank_income['2015 [YR2015]'] != '. .']
    # quick merge
    # find matching indices 
    shared_index = [index for index in oecd_income.index if index in world_bank_income.index]
    
    # select data
    oecd_income_subset = oecd_income.loc[shared_index]
    world_bank_income_subset = world_bank_income.loc[shared_index]
    oecd_income_subset['WorldBank'] = np.float32(world_bank_income_subset['2015 [YR2015]'])
    oecd_income_subset['Income'] = np.float32(oecd_income_subset['Income'])

    # fit linear model
    model = LinearRegression(fit_intercept=False)
    x = np.array(oecd_income_subset['Income']).reshape((-1, 1))
    y = np.array(oecd_income_subset['WorldBank'])
    model.fit(x, y)
    r_sq = model.score(x, y)
    print(f"coefficient of determination: {r_sq}")
    print(f"intercept: {model.intercept_}")
    print(f"slope: {model.coef_}")
    
    # plot result
    x_line = np.arange(0, 70000)
    y_pred = model.predict(x_line.reshape((-1, 1)))

    fig, ax = plt.subplots()
    ax.plot(x_line, y_pred, linestyle = '--', color = 'k')
    ax.scatter(x, y)
    ax.set_xlim([0, 70000])
    ax.set_ylim([0, 70000])
    ax.set_xlabel('OECD stats disposable income')
    ax.set_ylabel('WorldBank adjusted GDPpc')
    rsqrt = str(np.round(model.coef_[0],2))
    ax.set_title(f'Rsqrt: {rsqrt}')
    plt.show()








if __name__ == '__main__':
    merge_OECD_WorldBankData()