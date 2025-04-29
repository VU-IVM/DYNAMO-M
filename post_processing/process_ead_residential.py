import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def export_region_ead_residential(path, area, behaves, rcp, plot=True, n_runs=None):
    if rcp == 'rcp4p5':
        ssp = 'ssp2'
    elif rcp == 'rcp8p5':
        ssp = 'ssp5'
    else:
        raise ValueError('rcp not recognized')

    for behave in behaves:
        results_control = pd.DataFrame()
        results_rcp = pd.DataFrame()

        # get path from multirun to load individual runs
        path_control = os.path.join(path, behave, area, f'control_{ssp}', 'individual_runs')
        path_rcp = os.path.join(path, behave, area, f'{rcp}_{ssp}', 'individual_runs')
        if not os.path.exists(path_control) or not os.path.exists(path_rcp):
            print(f'Path does not exist: {path_control} or {path_rcp}')
            continue
        if n_runs == None:
            runs = os.listdir(path_control)
            # get run ids
            ids = np.arange(np.max([int(run[-2:]) for run in runs])+1)
            base = 'run_'
            runs_sorted = [base + str(id).zfill(2) for id in ids]
        else:
            runs_sorted = [f'run_{i:02d}' for i in range(n_runs)]

        # iterate over runs and store individual countries
        for run_folder in runs_sorted:
            path_run_control = os.path.join(path_control, run_folder, 'ead_residential_nodes.csv')
            path_run_rcp = os.path.join(path_rcp, run_folder, 'ead_residential_nodes.csv')

            if (os.path.exists(path_run_control) and os.path.exists(path_run_rcp)):

                # check if merged filename exists, if so, use that one
                if os.path.exists(path_run_control.replace('.csv', '_merged.csv')):
                    control_ead = pd.read_csv(path_run_control.replace('.csv', '_merged.csv'), index_col=0)
                    rcp_ead = pd.read_csv(path_run_rcp.replace('.csv', '_merged.csv'), index_col=0)
                    print('using merged files [export_region_ead]')

                else:
                    # load population in floodplain
                    control_ead = pd.read_csv(path_run_control, index_col=0)
                    rcp_ead = pd.read_csv(path_run_rcp, index_col=0)

                # calculate migration and store
                results_control[run_folder] = control_ead.sum(axis=1)
                results_rcp[run_folder] = rcp_ead.sum(axis=1)
            else:
                print(f'Path does not exist: {path_run_control} or {path_run_rcp}')

        # calculate some statistics 
        statistics_control = pd.DataFrame()
        statistics_control['minimum'] = results_control.min(axis = 1)
        statistics_control['mean'] = results_control.mean(axis = 1)
        statistics_control['maximum'] = results_control.max(axis = 1)

        # calculate for rcp 
        statistics_rcp = pd.DataFrame()
        statistics_rcp['minimum'] = results_rcp.min(axis = 1)
        statistics_rcp['mean'] = results_rcp.mean(axis = 1)
        statistics_rcp['maximum'] = results_rcp.max(axis = 1)



        # # export all
        path_for_export = os.path.join('DataDrive', 'processed_results', area, behave, 'risk')
        os.makedirs(path_for_export, exist_ok=True)
        statistics_control.to_csv(os.path.join(path_for_export, f'ead_residential_control_{ssp}_{area}.csv'))
        statistics_rcp.to_csv(os.path.join(path_for_export, f'ead_residential_{rcp}_{ssp}_{area}.csv'))

        if plot:
            # remove spinup from plot
            to_keep = [row for row in statistics_rcp.index if not row.startswith('spin')]
            statistics = statistics_rcp.loc[to_keep]
            # inlcude plot for easy comparisons
            fig, ax = plt.subplots()
            x = np.arange(2015, 2015 + len(statistics['minimum']))        
            ax.fill_between(x, np.array(statistics['minimum']), np.array(statistics['maximum']), color = 'grey', alpha = 0.2)
            ax.plot(x, np.array(statistics['mean']), color = 'k')
            if not statistics['minimum'].min() == statistics['maximum'].max():
                try:
                    ax.set_ylim([statistics['minimum'].min(), statistics['maximum'].max()])
                except:
                    print('Plotting ead failed')
            ax.set_title(f'{area}_{behave}_{rcp}')
            ax.tick_params(axis='x', labelrotation=90)
            fig.savefig(os.path.join(path_for_export, f'min_mean_max_ead_residential_{area}_{rcp}.png'))
            plt.close()
            del fig, ax

def export_country_ead_residential(path, area, behaves, rcp, plot=True, n_runs=None):
    if rcp == 'rcp4p5':
        ssp = 'ssp2'
    elif rcp == 'rcp8p5':
        ssp = 'ssp5'
    else:
        raise ValueError('rcp not recognized')

    NPV_ead = {}

    for behave in behaves:
        results = {}
        # get path from multirun to load individual runs
        path_rcp = os.path.join(path, behave, area, f'{rcp}_{ssp}', 'individual_runs')
        if n_runs == None:
            runs = os.listdir(path_rcp)
            # get run ids
            ids = np.arange(np.max([int(run[-2:]) for run in runs])+1)
            base = 'run_'
            runs_sorted = [base + str(id).zfill(2) for id in ids]
        else:
            runs_sorted = [f'run_{i:02d}' for i in range(n_runs)]

        # iterate over runs and store individual countries
        for run_folder in runs_sorted:
            path_run_rcp = os.path.join(path_rcp, run_folder, 'ead_residential_nodes.csv')

            if os.path.exists(path_run_rcp):

                # load population in floodplain
                # check if merged filename exists, if so, use that one
                if os.path.exists(path_run_rcp.replace('.csv', '_merged.csv')):
                    rcp_ead_run = pd.read_csv(path_run_rcp.replace('.csv', '_merged.csv'), index_col=0)
                    print(f'using merged files [export_country_migration]')
                else:
                    rcp_ead_run = pd.read_csv(path_run_rcp, index_col=0)

                # iterate over countries and only consider floodplains
                for country in np.unique([region[:3] for region in rcp_ead_run.columns]):
                    # subset dataframe based on country and floodplain
                    cols = [col for col in rcp_ead_run.columns if col.endswith('flood_plain') and col.startswith(country)]
                    rcp_ead_country_floodplain = rcp_ead_run[cols]

                    # for now just take ead
                    rcp_ead_country_floodplain_summed = rcp_ead_country_floodplain.sum(axis=1)
                    if country not in results.keys():
                        results[country] = {}
                        results[country][run_folder] = rcp_ead_country_floodplain_summed

                    else:
                        results[country][run_folder] = rcp_ead_country_floodplain_summed

        # now iterate over countries and export migration of individual runs
        for country in results.keys():
            country_df = pd.DataFrame()

            for run in results[country].keys():
                country_df[run] = results[country][run]
            
            # do some quick statistics to summarize
            statistics = pd.DataFrame()
            statistics['minimum'] = country_df.min(axis = 1)
            statistics['mean'] = country_df.mean(axis = 1)
            statistics['maximum'] = country_df.max(axis = 1)
         
            # export
            path_for_export = os.path.join('DataDrive', 'processed_results', area, 'individual_countries', country, behave, 'risk')
            os.makedirs(path_for_export, exist_ok=True)
            country_df.to_csv(os.path.join(path_for_export, f'ead_residential_{rcp}_{country}.csv'))
            statistics.to_csv(os.path.join(path_for_export, f'min_mean_max_ead_residential_{rcp}_{country}.csv'))
            
            # export NPV to table
            discount_rate=0.04

            for stat in ['mean']:
                array_spendings = np.array(statistics[stat].loc[[index for index in statistics.index if not index.startswith('spin')]])
                cumsum = np.cumsum(array_spendings)
                discounts = 1 / (1 + discount_rate)**np.arange(cumsum.size)
                NPV = np.sum(cumsum * discounts)
                # if behave == 'no_adaptation': NPV = 0 # why is this only happening in multiruns?
                if not behave in NPV_ead.keys():
                    NPV_ead[behave] = {}
                NPV_ead[behave][country] = NPV



            # export relative migration
            if plot:
                # remove spinup from plot
                to_keep = [row for row in statistics.index if not row.startswith('spin')]
                statistics = statistics.loc[to_keep]
                # inlcude plot for easy comparisons
                fig, ax = plt.subplots()
                x = np.arange(2015, 2015 + len(statistics['minimum']))        
                ax.fill_between(x, np.array(statistics['minimum']), np.array(statistics['maximum']), color = 'grey', alpha = 0.2)
                ax.plot(x, np.array(statistics['mean']), color = 'k')
                if not statistics['minimum'].min() == statistics['maximum'].max():
                    ax.set_ylim([statistics['minimum'].min(), statistics['maximum'].max()])
                ax.set_title(f'{country}_{behave}_{rcp}')
                ax.tick_params(axis='x', labelrotation=90)
                fig.savefig(os.path.join(path_for_export, f'min_mean_max_{country}_{rcp}.png'))
                plt.close()
                del fig, ax
                
    # export NPV to csv

if __name__ == '__main__':
    path_multiruns = os.path.join('DataDrive', 'MULTIRUNS_GLOBAL')
    path_processed_results = os.path.join('DataDrive', 'processed_results')
    n_runs = 25

    if os.path.exists(path_processed_results):
        print('Already processed results')
        # return
    behaves = ['proactive_government', 'maintain_fps', 'no_government', 'no_adaptation']
   
    areas = ['asia', 'africa', 'europe', 'oceania', 'south_america', 'central_america', 'northern_america']
    rcps = ['rcp4p5', 'rcp8p5']
    for area in areas:
        for rcp in rcps:
            print('Processing:', area, rcp)
            export_country_ead_residential(path_multiruns, area, behaves, rcp, n_runs=n_runs)
            export_region_ead_residential(path_multiruns, area, behaves, rcp, n_runs=n_runs)
