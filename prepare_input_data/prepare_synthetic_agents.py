import pandas as pd
import numpy as np
import os
import geopandas as gpd
from scipy import stats
from prepare_income_data import prepare_income_data_synthpop, rasterize_income_data
import gzip

def read_synthetic_population_data(
    data_folder = os.path.join('DataDrive'),
    n_columns = None,
    default_ratio = 1.15):

    # column names data for easy reference
    colnames = ['INCOME', 'RURAL', 'FARMING', 'AGECAT', 'GENDER', 
                            'EDUCAT', 'HHTYPE', 'HHID', 'RELATE', 'HHSIZECAT', 'GEOLEV1']

    # iterate of files in datafolder and read
    agent_folder = r'/scistor/ivm/data_catalogue/population/GLOPOP-S'
    files = os.listdir(agent_folder)  
    # get .dat files only
    agent_data = [file for file in files if file.endswith('.dat.gz')]
    
    # get indidual countries on disk
    countries_on_disk = np.unique([file[:3] for file in agent_data])

    # load geoms for sampling region names
    geoms_gdl = gpd.read_file(os.path.join(data_folder, 'SLR', 'GDL', 'GDL.shp'))

    # load income per capita world bank
    income_fn = os.path.join(data_folder, 'ECONOMY', 'adjusted_net_national_income_pc_WoldBank_filled.csv')
    df = pd.read_csv(income_fn)[['Country Code', '2015 [YR2015]']].set_index('Country Code', drop=True).dropna()
    df.drop(df[df['2015 [YR2015]'] == '..'].index, inplace = True)
    df['2015 [YR2015]'] = np.array( df['2015 [YR2015]'], np.float32)
    income = df

    # load preprocessed income data OECD for ratios
    mean_median_income_ratios_fp = os.path.join(data_folder, 'ECONOMY', 'PROCESSED', 'mean_median_WIID.csv')
    income_ratios = pd.read_csv(mean_median_income_ratios_fp, index_col=0)

    # load translaters
    # intitate
    max_household_size = 0

    # create dict for summary
    ids = []
    gdlcodes = []
    population = []
    n_households = []
    median_inc_perc = []
    mean_median_ratios = []
    dom_hh_type = []
    mean_hh_size = []
    rel_income_percentile = []
    average_income_region = []
    

    # for results
    gadm_regions_not_found = []
    h = 0

    # iterate over rows GDL
    for _, geom in geoms_gdl.iterrows():
        GDLcode = geom['gdlcode']
        iso3 = geom['iso_code']
        # account for disputed territories in gdl dataset (e.g. Sahel in Marocco)
        if iso3 == 'NA':
            iso3 = GDLcode[:3] 
        ID = str(geom['ID'])

        if iso3 in countries_on_disk:
            # subset income data
            if iso3 in income.index:
                # check if ratio is available else use default
                if iso3 in income_ratios.index:
                    mean_median_inc_ratio = income_ratios.loc[iso3]['mean_median_ratio']
                else:
                    mean_median_inc_ratio = default_ratio

                average_income = income.loc[iso3].iloc[0]
                median_income = average_income/ mean_median_inc_ratio
                # mean_income = self.income_region * \
                #     self.model.settings['adaptation']['mean_median_inc_ratio']
                mu = np.log(median_income)
                sd = np.sqrt(2 * np.log(average_income / median_income))
                national_income_distribution = np.sort(
                    np.random.lognormal(
                        mu, sd, 20_000).astype(
                        np.int32)) 
            else:
                national_income_distribution = np.zeros(100)
                print(f'no income found for {iso3}')
            
            # find file to load
            filename = [fn for fn in agent_data if GDLcode in fn]
            if len(filename) > 0:
                filename = filename[0]

                # remove filename from list of agents to load
                agent_data.remove(filename)
                # load and reshape data
                data_fn = os.path.join(agent_folder, filename)
                
                # load
                with gzip.open(data_fn, 'rb') as f:
                    data_np = np.frombuffer(f.read(), dtype=np.int32)

                # reshapa data
                n_columns = 10
                try:
                    # get individuals
                    n_people = data_np.size// n_columns
                    assert n_people == data_np.size/ n_columns
                    data_reshaped = np.reshape(data_np, (n_columns, n_people)).transpose()

                except:
                    try:
                        n_columns = 11
                        n_people = data_np.size// n_columns
                        assert n_people == data_np.size/ n_columns
                        data_reshaped = np.reshape(data_np, (n_columns, n_people)).transpose()
                    except:
                        print(f'reshape failed for {filename}, datasize: {data_np.size}')
                        continue
                
                # Extract only households
                HHID, indices, counts = np.unique(data_reshaped[:, 7], return_index=True, return_counts=True)
                
                ###################################
                ###### export household sizes #####
                ###################################
                # get actual household sizes per houshold
                household_sizes = counts
                assert counts.size == indices.size

                # cap household size to 15 individuals to keep redundancy arrays within manageable bounds
                # if household_sizes.max() > 15:
                    # print(f'Household size of {household_sizes.max()} encountered in {iso3}. Capped to 15.')
                # update max household size
                household_sizes = np.minimum(household_sizes, 15)
                
                max_household_size = np.maximum(max_household_size, household_sizes.max())
                assert all(household_sizes != -1)

                ######################################
                ######## Export household type #######
                ######################################

                household_types = data_reshaped[indices, 6]                    
                
                ######################################
                ###### export income percentiles #####
                ######################################

                income_classes_households = data_reshaped[indices, 0]
                income_percentiles_households = np.full(income_classes_households.size, -1, np.int32)

                # generate and assign
                income_percentiles_households[income_classes_households==1] = np.random.randint(0, 21, income_classes_households[income_classes_households==1].size)
                income_percentiles_households[income_classes_households==2] = np.random.randint(20, 41, income_classes_households[income_classes_households==2].size)
                income_percentiles_households[income_classes_households==3] = np.random.randint(40, 61, income_classes_households[income_classes_households==3].size)
                income_percentiles_households[income_classes_households==4] = np.random.randint(60, 81, income_classes_households[income_classes_households==4].size)
                income_percentiles_households[income_classes_households==5] = np.random.randint(80, 100, income_classes_households[income_classes_households==5].size)
                if not all(income_percentiles_households != -1):
                    print(f'Failed for {filename}')
                    continue

                sorted_income_percentiles = np.sort(income_percentiles_households)
                percentile, counts = np.unique(sorted_income_percentiles, return_counts = True)
                # calculate count percentiles
                cummulative_counts = np.full(counts.size, -1, np.int32)
                
                try:
                    for i, count in enumerate(counts):
                        cummulative_counts[i] = counts[:i+1].sum()/ counts.sum() * 100
                    relative_income_percentile = np.take(cummulative_counts, income_percentiles_households)
                except:
                    print(f'failed for {GDLcode}')
                    continue
                # since the 100th percentile does not exist, cap at 99
                relative_income_percentile = np.minimum(relative_income_percentile, 99)

                    


                #############################################
                ### export in folders relative to gdl ID ####
                #############################################
                folder_export = os.path.join(data_folder, 'AGENTS', 'PROCESSED', f'r{ID.zfill(4)}')
                if not os.path.exists(folder_export):
                    os.makedirs(folder_export)
                fn_size = os.path.join(folder_export, 'sizes_region.npy')
                fn_household_type = os.path.join(folder_export, 'household_type_region.npy')
                fn_income_perc = os.path.join(folder_export, 'income_perc_region.npy')

                np.save(fn_size, household_sizes)
                np.save(fn_household_type, household_types)
                np.save(fn_income_perc, relative_income_percentile)

                # Summarize:
                mean_hh_size.append(np.round(household_sizes.mean(), 2))
                gdlcodes.append(GDLcode)
                ids.append(ID)
                mean_median_ratios.append(mean_median_inc_ratio)
                population.append(np.sum(household_sizes))
                n_households.append(household_sizes.size)
                dom_hh_type.append(stats.mode(household_types, keepdims=False).mode)
                median_inc_perc.append(round(np.median(income_percentiles_households), 2))
                rel_income_percentile.append(round(np.median(relative_income_percentile), 2))
                average_income_region.append(round(np.percentile(national_income_distribution, income_percentiles_households).mean()))
            else:
                gadm_regions_not_found.append(GDLcode)

        print(f'{h} of {len(geoms_gdl)} ({GDLcode})')
        h+=1

        # store in pandas and save
        summary_pd = pd.DataFrame({
            'ID': ids,
            'GLDcode': gdlcodes,
            'population': population,
            'n_households': n_households,
            'dominant_hh_type': dom_hh_type,
            'median_inc_perc': median_inc_perc,
            'median_rel_inc_perc': rel_income_percentile,   
            'mean_hh_size': mean_hh_size,
            'average_income_region': average_income_region,
            'mean_median_inc_ratio': mean_median_ratios
            })
        
        summary_pd.to_csv(os.path.join(data_folder, 'AGENTS', 'PROCESSED', 'summary.csv'), index=False)
    
    # export files that were not loaded
    pd.DataFrame({'agents_not_assigned_to_region': np.array(agent_data)}).to_csv(os.path.join(data_folder, 'AGENTS', 'PROCESSED', 'agents_not_assigned.csv'), index=False)
    pd.DataFrame({'gdl_region_not_in_synthpop': np.array(gadm_regions_not_found)}).to_csv(os.path.join(data_folder, 'AGENTS', 'PROCESSED', 'gdl_not_in_synthpop.csv'), index=False)


    # rasterize income data
    print('rasterizing income data')
    processed_income_fp = prepare_income_data_synthpop(os.path.join('DataDrive', 'AGENTS', 'PROCESSED'))
    rasterize_income_data(processed_income_fp)
    return max_household_size

if __name__ == '__main__':
    max_household_size = read_synthetic_population_data()
    print(f'max household size: {max_household_size}')