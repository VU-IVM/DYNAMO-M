import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

def export_n_persons_moving_in_and_out(path, area, behaves, rcp, plot=True, export_runs=False, n_runs=None):
    if rcp == 'rcp4p5':
        ssp = 'ssp2'
    elif rcp == 'rcp8p5':
        ssp = 'ssp5'
    else:
        raise ValueError('rcp not recognized')

    for behave in behaves:
        results_moving_in_control = pd.DataFrame()
        results_moving_out_control = pd.DataFrame()
        results_moving_in_rcp = pd.DataFrame()
        results_moving_out_rcp = pd.DataFrame()

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
            load_file = os.path.join(path_control, run_folder, 'n_persons_moving_in.csv')
            if os.path.exists(load_file.replace('.csv', '_merged.csv')):
                # load control
                path_run_control_moving_in = os.path.join(path_control, run_folder, 'n_persons_moving_in.csv').replace('.csv', '_merged.csv')
                path_run_control_moving_out = os.path.join(path_control, run_folder, 'n_persons_moving_out.csv').replace('.csv', '_merged.csv')
                control_moving_in = pd.read_csv(path_run_control_moving_in, index_col=0)
                control_moving_out = pd.read_csv(path_run_control_moving_out, index_col=0)
                
                # load rcp
                path_run_rcp_moving_in = os.path.join(path_rcp, run_folder, 'n_persons_moving_in.csv').replace('.csv', '_merged.csv')
                path_run_rcp_moving_out = os.path.join(path_rcp, run_folder, 'n_persons_moving_out.csv').replace('.csv', '_merged.csv')
                rcp_moving_in = pd.read_csv(path_run_rcp_moving_in, index_col=0)
                rcp_moving_out = pd.read_csv(path_run_rcp_moving_out, index_col=0)
                print('using merged files [export_n_persons_moving_in_and_out]')

            else:

                # load control
                path_run_control_moving_in = os.path.join(path_control, run_folder, 'n_persons_moving_in.csv')
                path_run_control_moving_out = os.path.join(path_control, run_folder, 'n_persons_moving_out.csv')
                control_moving_in = pd.read_csv(path_run_control_moving_in, index_col=0)
                control_moving_out = pd.read_csv(path_run_control_moving_out, index_col=0)
                
                # load rcp
                path_run_rcp_moving_in = os.path.join(path_rcp, run_folder, 'n_persons_moving_in.csv')
                path_run_rcp_moving_out = os.path.join(path_rcp, run_folder, 'n_persons_moving_out.csv')
                rcp_moving_in = pd.read_csv(path_run_rcp_moving_in, index_col=0)
                rcp_moving_out = pd.read_csv(path_run_rcp_moving_out, index_col=0)

            # remove spinup
            control_moving_in = control_moving_in.loc[control_moving_in.index.str.contains('spin') == False]
            control_moving_out = control_moving_out.loc[control_moving_out.index.str.contains('spin') == False]
            rcp_moving_in = rcp_moving_in.loc[rcp_moving_in.index.str.contains('spin') == False]   
            rcp_moving_out = rcp_moving_out.loc[rcp_moving_out.index.str.contains('spin') == False]

            # sum over simulation period
            control_moving_in_sum = control_moving_in.sum(axis=0)
            control_moving_out_sum = control_moving_out.sum(axis=0)
            rcp_moving_in_sum = rcp_moving_in.sum(axis=0)
            rcp_moving_out_sum = rcp_moving_out.sum(axis=0)

            # store in df to later calculate average over runs
            results_moving_in_control[run_folder] = control_moving_in_sum
            results_moving_out_control[run_folder] = control_moving_out_sum
            results_moving_in_rcp[run_folder] = rcp_moving_in_sum
            results_moving_out_rcp[run_folder] = rcp_moving_out_sum

        if export_runs:
            # export all
            path_for_export = os.path.join('DataDrive', 'processed_results', area, behave, 'n_persons_moving')
            os.makedirs(path_for_export, exist_ok=True)
            results_moving_in_control.to_csv(os.path.join(path_for_export, f'all_moving_in_control_{rcp}_{area}.csv'))
            results_moving_out_control.to_csv(os.path.join(path_for_export, f'all_moving_out_control_{rcp}_{area}.csv'))
            results_moving_in_rcp.to_csv(os.path.join(path_for_export, f'all_moving_in_{rcp}_{area}.csv'))
            results_moving_out_rcp.to_csv(os.path.join(path_for_export, f'all_moving_out_{rcp}_{area}.csv'))

        # average over runs
        averages = pd.DataFrame()
        averages['moving_in_control'] = results_moving_in_control.mean(axis=1, skipna=True)
        averages['moving_out_control'] = results_moving_out_control.mean(axis=1, skipna=True) 
        averages['moving_in_rcp'] = results_moving_in_rcp.mean(axis=1, skipna=True)
        averages['moving_out_rcp'] = results_moving_out_rcp.mean(axis=1, skipna=True)
        averages['moving_in_normalized'] = averages['moving_in_rcp'] / averages['moving_in_control']
        averages['moving_out_normalized'] = averages['moving_out_rcp'] / averages['moving_out_control']
        # export averages
        path_for_export = os.path.join('DataDrive', 'processed_results', area, behave, 'n_persons_moving')
        os.makedirs(path_for_export, exist_ok=True)
        averages.to_csv(os.path.join(path_for_export, f'averages_moving_{rcp}_{area}.csv'))


def export_regions_migration(path, area, behaves, rcp, plot=True, n_runs=None):
    if rcp == 'rcp4p5':
        ssp = 'ssp2'
    elif rcp == 'rcp8p5':
        ssp = 'ssp5'
    else:
        raise ValueError('rcp not recognized')

    for behave in behaves:
        results = pd.DataFrame()
        population_fp_control = pd.DataFrame()
        population_fp_rcp = pd.DataFrame()

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
        using_merged_files = False
        for run_folder in runs_sorted:
            path_run_control = os.path.join(path_control, run_folder, 'n_persons_moving_out.csv')
            path_run_rcp = os.path.join(path_rcp, run_folder, 'n_persons_moving_out.csv')
            path_run_pop_control = os.path.join(path_control, run_folder, 'population.csv')
            path_run_pop_rcp = os.path.join(path_rcp, run_folder, 'population.csv')

            if (os.path.exists(path_run_control) and os.path.exists(path_run_rcp)):

                # load population in 
                # check if merged filename exists, if so, use that one
                if os.path.exists(path_run_control.replace('.csv', '_merged.csv')) or using_merged_files:
                    control_n_out_country_floodplain = pd.read_csv(path_run_control.replace('.csv', '_merged.csv'), index_col=0)
                    rcp_n_out_country_floodplain = pd.read_csv(path_run_rcp.replace('.csv', '_merged.csv'), index_col=0)
                    control_population_country_floodplain = pd.read_csv(path_run_pop_control.replace('.csv', '_merged.csv'), index_col=0).dropna(axis=1)
                    rcp_population_country_floodplain = pd.read_csv(path_run_pop_rcp.replace('.csv', '_merged.csv'), index_col=0).dropna(axis=1)
                    using_merged_files = True # to make use we try to keep using merged files
                    # print('using merged files [export_regions_migration]')
                    
                else:
                    control_n_out_country_floodplain = pd.read_csv(path_run_control, index_col=0)
                    rcp_n_out_country_floodplain = pd.read_csv(path_run_rcp, index_col=0)
                    control_population_country_floodplain = pd.read_csv(path_run_pop_control, index_col=0)
                    rcp_population_country_floodplain = pd.read_csv(path_run_pop_rcp, index_col=0)

                # sum over all floodplains
                floodplains_control = [col for col in control_n_out_country_floodplain.columns if col.endswith('flood_plain')]
                floodplains_rcp = [col for col in rcp_n_out_country_floodplain.columns if col.endswith('flood_plain')]
                control_n_out_country_floodplain = control_n_out_country_floodplain[floodplains_control].sum(axis=1)
                rcp_n_out_country_floodplain = rcp_n_out_country_floodplain[floodplains_rcp].sum(axis=1)

                # calculate migration and store
                migration = rcp_n_out_country_floodplain - control_n_out_country_floodplain
                migration = migration.cumsum()
                
                assert migration[:15].sum() == 0, f'{path_run_control} and {path_run_rcp} do not match'
 
                results[run_folder] = migration
                
                # also store population in floodplain country
                population_fp_control[run_folder] = control_population_country_floodplain.filter(regex='flood_plain$', axis=1).sum(axis=1)

                population_fp_rcp[run_folder] = rcp_population_country_floodplain.filter(regex='flood_plain$', axis=1).sum(axis=1)
            else:
                print(f'Path does not exist: {path_run_control} or {path_run_rcp}')
                # return None
        # calculate some statistics on migration
        statistics = pd.DataFrame()
        statistics['minimum'] = results.min(axis = 1)
        statistics['mean'] = results.mean(axis = 1)
        statistics['maximum'] = results.max(axis = 1)

        # export all
        path_for_export = os.path.join('DataDrive', 'processed_results', area, behave, 'migration')
        os.makedirs(path_for_export, exist_ok=True)
        results.to_csv(os.path.join(path_for_export, f'all_migration_{rcp}_{area}.csv'))
        statistics.to_csv(os.path.join(path_for_export, f'min_mean_max_migration_{rcp}_{area}.csv'))

        if plot:
            # remove spinup from plot
            to_keep = [row for row in statistics.index if not row.startswith('spin')]
            test = statistics.copy()
            statistics = statistics.loc[to_keep]
            # inlcude plot for easy comparisons
            fig, ax = plt.subplots()
            x = np.arange(2015, 2015 + len(statistics['minimum']))        
            ax.fill_between(x, np.array(statistics['minimum']), np.array(statistics['maximum']), color = 'grey', alpha = 0.2)
            ax.plot(x, np.array(statistics['mean']), color = 'k')
            if not statistics['minimum'].min() == statistics['maximum'].max():
                try:
                    ax.set_ylim([statistics['minimum'].min(), statistics['maximum'].max()])
                except:
                    print('plotting migration failed')
            ax.set_title(f'{area}_{behave}_{rcp}')
            ax.tick_params(axis='x', labelrotation=90)
            fig.savefig(os.path.join(path_for_export, f'min_mean_max_{area}_{rcp}.png'))
            plt.close()
            del fig, ax

        # also do some statistics on population
        # process relative migration and export as well
        statistics_relative = pd.DataFrame()
        relative_migration = results/ population_fp_control * 100
        statistics_relative['minimum'] = relative_migration.min(axis=1)
        statistics_relative['mean'] = relative_migration.mean(axis=1)
        statistics_relative['maximum'] = relative_migration.max(axis=1)
        statistics_relative.to_csv(os.path.join(path_for_export, f'relative_migration_{rcp}_{area}.csv'))
       
        # also export population in floodplain
        path_for_export = os.path.join('DataDrive', 'processed_results', area, behave, 'population')
        os.makedirs(path_for_export, exist_ok=True)
        population_fp_control.mean(axis=1).to_csv(os.path.join(path_for_export, f'pop_fp_control_{ssp}_{area}.csv'))
        population_fp_rcp.mean(axis=1).to_csv(os.path.join(path_for_export, f'pop_fp_{rcp}_{ssp}_{area}.csv'))

def export_country_migration(path, area, behaves, rcp, plot=True, n_runs=None):
    if rcp == 'rcp4p5':
        ssp = 'ssp2'
    elif rcp == 'rcp8p5':
        ssp = 'ssp5'
    else:
        raise ValueError('rcp not recognized')

    for behave in behaves:
        migration_per_gdl = {}
        results = {}
        population_fp = {}
        population_per_gdl = {}
        # get path from multirun to load individual runs
        path_control = os.path.join(path, behave, area, f'control_{ssp}', 'individual_runs')
        path_rcp = os.path.join(path, behave, area, f'{rcp}_{ssp}', 'individual_runs')
        assert os.path.exists(path_control) and os.path.exists(path_rcp), f'Path does not exist: {path_control} or {path_rcp}'
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
            path_run_control = os.path.join(path_control, run_folder, 'n_persons_moving_out.csv')
            path_run_rcp = os.path.join(path_rcp, run_folder, 'n_persons_moving_out.csv')
            path_run_pop_control = os.path.join(path_control, run_folder, 'population.csv')
            path_run_pop_rcp = os.path.join(path_rcp, run_folder, 'population.csv')

            assert(os.path.exists(path_run_control) and os.path.exists(path_run_rcp))

            # load population in floodplain
            # check if merged filename exists, if so, use that one
            if os.path.exists(path_run_control.replace('.csv', '_merged.csv')):
                control_n_out_floodplain_run = pd.read_csv(path_run_control.replace('.csv', '_merged.csv'), index_col=0)
                rcp_n_out_floodplain_run = pd.read_csv(path_run_rcp.replace('.csv', '_merged.csv'), index_col=0)
                control_population_floodplain = pd.read_csv(path_run_pop_control.replace('.csv', '_merged.csv'), index_col=0)
                rcp_population_floodplain = pd.read_csv(path_run_pop_rcp.replace('.csv', '_merged.csv'), index_col=0)                    
                # print(f'using merged files [export_country_migration]')
            else:
                control_n_out_floodplain_run = pd.read_csv(path_run_control, index_col=0)
                rcp_n_out_floodplain_run = pd.read_csv(path_run_rcp, index_col=0)
                control_population_floodplain = pd.read_csv(path_run_pop_control, index_col=0)
                rcp_population_floodplain = pd.read_csv(path_run_pop_rcp, index_col=0)

            # iterate over countries and only consider floodplains
            for country in np.unique([region[:3] for region in rcp_n_out_floodplain_run.columns]):
                # subset dataframe based on country and floodplain
                cols_control = [col for col in control_n_out_floodplain_run.columns if col.endswith('flood_plain') and col.startswith(country)]
                cols_rcp = [col for col in rcp_n_out_floodplain_run.columns if col.endswith('flood_plain') and col.startswith(country)]
                control_n_out_floodplain_run_country = control_n_out_floodplain_run[cols_control]
                rcp_n_out_floodplain_run_country = rcp_n_out_floodplain_run[cols_rcp]
                control_population_country_floodplain = control_population_floodplain[cols_control]
                rcp_population_country_floodplain = rcp_population_floodplain[cols_rcp]

                # could export if wanted for individual floodplains
                # check if cols control and rcp are the same
                for col in cols_control:
                    if col not in cols_rcp:
                        rcp_n_out_floodplain_run_country[col] = 0
                for col in cols_rcp:
                    if col not in cols_control:
                        control_n_out_floodplain_run_country[col] = 0

                migration = rcp_n_out_floodplain_run_country - control_n_out_floodplain_run_country
                migration = migration.cumsum(axis=0)
                
                # for now just sum whole country
                migration_summed = migration.sum(axis=1)
                if country not in results.keys():
                    results[country] = {}
                    results[country][run_folder] = migration_summed
                    migration_per_gdl[country] = {}
                    migration_per_gdl[country][run_folder] = migration
                    population_fp[country] = {}
                    population_fp[country][run_folder] = {}
                    population_per_gdl[country] = {}
                    population_per_gdl[country][run_folder] = {}
                    population_fp[country][run_folder]['control'] = control_population_country_floodplain.sum(axis=1)
                    population_fp[country][run_folder][rcp] = rcp_population_country_floodplain.sum(axis=1)
                    population_per_gdl[country][run_folder]['control'] = control_population_country_floodplain
                    population_per_gdl[country][run_folder][rcp] = rcp_population_country_floodplain
                else:
                    results[country][run_folder] = migration_summed
                    migration_per_gdl[country][run_folder] = migration
                    population_fp[country][run_folder] = {}
                    population_fp[country][run_folder]['control'] =control_population_country_floodplain.sum(axis=1)
                    population_fp[country][run_folder][rcp] = rcp_population_country_floodplain.sum(axis=1)
                    population_per_gdl[country][run_folder] = {}
                    population_per_gdl[country][run_folder]['control'] =control_population_country_floodplain
                    population_per_gdl[country][run_folder][rcp] = rcp_population_country_floodplain

        # now iterate over countries and export migration of individual runs
        for country in results.keys():
            country_df = pd.DataFrame()
            migration_per_gdl_df = pd.DataFrame()
            population_control_df = pd.DataFrame()
            population_rcp_df = pd.DataFrame()
            population_control_per_gdl_df = pd.DataFrame()
            population_rcp_per_gdl_df = pd.DataFrame()

            for run in results[country].keys():
                country_df[run] = results[country][run]
                migration_per_gdl_df = pd.concat([migration_per_gdl_df, migration_per_gdl[country][run]], axis=0)
                population_control_df[run] = population_fp[country][run]['control']
                population_rcp_df[run] = population_fp[country][run][rcp]
                population_control_per_gdl_df = pd.concat([population_control_per_gdl_df, population_per_gdl[country][run]['control']], axis=0)
                population_rcp_per_gdl_df = pd.concat([population_rcp_per_gdl_df, population_per_gdl[country][run][rcp]], axis=0)
            
            # do some quick statistics to summarize
            statistics = pd.DataFrame()
            statistics['minimum'] = country_df.min(axis = 1)
            statistics['mean'] = country_df.mean(axis = 1)
            statistics['maximum'] = country_df.max(axis = 1)

            # take the average of all runs for gdl regions in country
            average_migration_gdl_regions = migration_per_gdl_df.groupby(migration_per_gdl_df.index).mean().loc[statistics.index]

            # also take average for populations
            average_population_gdl_regions_control = population_control_per_gdl_df.groupby(population_control_per_gdl_df.index).mean().loc[statistics.index]
            average_population_gdl_regions_rcp = population_rcp_per_gdl_df.groupby(population_rcp_per_gdl_df.index).mean().loc[statistics.index]

            # export
            path_for_export = os.path.join('DataDrive', 'processed_results', area, 'individual_countries', country, behave)
            os.makedirs(os.path.join(path_for_export, 'migration'), exist_ok=True)
            os.makedirs(os.path.join(path_for_export, 'population'), exist_ok=True)
            country_df.to_csv(os.path.join(path_for_export, f'migration_{rcp}_{country}.csv'))
            statistics.to_csv(os.path.join(path_for_export, f'min_mean_max_migration_{rcp}_{country}.csv'))
            average_migration_gdl_regions.to_csv(os.path.join(path_for_export, 'migration',  f'average_migration_gdl_regions_{rcp}_{country}.csv'))
            average_population_gdl_regions_control.to_csv(os.path.join(path_for_export, 'population', f'average_population_gdl_regions_control_{ssp}_{country}.csv'))
            average_population_gdl_regions_rcp.to_csv(os.path.join(path_for_export, 'population', f'average_population_gdl_regions_{rcp}_{ssp}_{country}.csv'))


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
                

            # also export population in floodplain
            path_for_export = os.path.join('DataDrive', 'processed_results', area, 'individual_countries', country, behave, 'population')
            os.makedirs(path_for_export, exist_ok=True)
            population_control_df.mean(axis=1).to_csv(os.path.join(path_for_export, f'pop_fp_control_{ssp}_{country}.csv'))
            population_rcp_df.mean(axis=1).to_csv(os.path.join(path_for_export, f'pop_fp_{rcp}_{ssp}_{country}.csv'))


if __name__ == '__main__':
    path = 'DataDrive/MULTIRUNS_GLOBAL'
    behaves = ['proactive_government', 'maintain_fps', 'no_government', 'no_adaptation']
    areas = ['south_america', 'northern_america', 'central_america', 'africa', 'oceania', 'europe', 'asia']
    areas = ['south_america', 'central_america']
    rcps = ['rcp4p5', 'rcp8p5']
    
    for area in areas:
        for rcp in rcps:
            export_country_migration(path, area, behaves, rcp, plot=False)
            export_regions_migration(path, area, behaves, rcp, plot=False)
            export_n_persons_moving_in_and_out(path, area, behaves, rcp, plot=False, export_runs=False)