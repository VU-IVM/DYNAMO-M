import numpy as np
import os
import pandas as pd
import geopandas as gpd
from honeybees.agents import AgentBaseClass
from scipy.spatial import distance_matrix as sdistance_matrix
from agents.coastal_nodes import CoastalNode
from agents.inland_nodes import InlandNode
from exporting.export_agents import export_agents, export_matrix
from agents.node_properties import NodeProperties
import geopandas as gpd


class Nodes(AgentBaseClass, NodeProperties):

    '''This class generates the inland and coastal nodes using the input data generated in prepare_input_data. 
    It generates  inland_node class instances for the inland admin regions, and for the part of each coastal admin region that lies outside of the floodplain.
    Coastal nodes are generated for the part of the coastal admin region that overlaps with the 1/100-year floodplain in 2080 under RCP 8.5.
    Inland nodes contain the aggregated households, whereas the coastal nodes contain the spatially explicit households.'''

    def __init__(self, model, agents):
        self.model = model
        self.agents = agents
        self.initiate_agents()
        self._load_initial_state()

    def initiate_agents(self):
        self.geoms = self.model.area.geoms['admin']
        self.n = len(self.geoms)
        self._initiate_attributes()

        max_household_size = 6
        
        self.household = {}
        self.aggregate_household = {}
        self.all_households = []
        j = 0
        n_households_per_region = np.full(len(self.geoms), 0, dtype=np.int32)
        for i, geom in enumerate(self.geoms):
            
            # All nodes
            ID = geom['properties']['id']

            # Coastal nodes 
            if ID.endswith('flood_plain') & self.model.config['general']['create_agents']:
                init_folder = os.path.join("DataDrive", "SLR", f"households_gadm_{self.model.args.admin_level}_{self.model.config['general']['start_time'].year}", self.model.config['general']['size'], ID.replace('_flood_plain', ''))
                locations_fn = os.path.join(init_folder, "locations.npy")
                if os.path.exists(locations_fn):
                    redundancy = max(int(np.load(locations_fn).size * .5), 100_000) # redundancy of 50%
                else:
                    redundancy = 0  # nobody should be moving to areas with no population
                household_class = CoastalNode(
                    self.model,
                    self.agents,
                    idx=i,
                    geom=geom,
                    distance_matrix=self.distance_matrix,
                    n_households_per_region=n_households_per_region,
                    init_folder=init_folder,
                    max_household_size=max_household_size,
                    redundancy=redundancy,
                    person_reduncancy=int(redundancy * (max_household_size // 2))
                )
                self.household[ID] = household_class
            else:

                household_class = InlandNode(
                    self.model,
                    self.agents,
                    idx=i,
                    geom=geom,
                    distance_matrix=self.distance_matrix,
                    n_households_per_region=n_households_per_region,
                    max_household_size=max_household_size
                )
                j += 1
                self.aggregate_household[ID] = household_class
            self.all_households.append(household_class)
        
        self.model.logger.info(f'Created {sum(household.n for household in self.household.values())} households')
        self.model.logger.info(f'Created {sum(household.n for household in self.aggregate_household.values())} aggregrate households')
        
    def _load_initial_state(self):
        return None

    def _get_distance_matrix(self):
        centroids = self.centroids
        
        # Convert (back) to geopandas points
        gpd_points = gpd.points_from_xy(centroids[:,0], centroids[:,1], crs='EPSG:4326')

        # Project points in world Mollweide (?)
        gpd_points = gpd_points.to_crs('ESRI:54009')

        # Extract x and y   
        x = gpd_points.x
        y = gpd_points.y

        # Stack into dataframe and export distance matrix
        projected_centroids = np.column_stack((x, y))
        return sdistance_matrix(projected_centroids, projected_centroids) /1000 # Devide by 1000 to derive km
    
    def _initiate_attributes(self):
        self.distance_matrix = self._get_distance_matrix()

    def merge_move_dictionary(self, move_dictionaries):
        merged_move_dictionary = {}
        for key in move_dictionaries[0].keys():
            # Household_ids are determined in the origin region. The procedure below ensures that
            # all household_ids are unique across the merged dictionary.
            if key == 'household_id':
                c = 0
                household_ids_per_region = [d[key] for d in move_dictionaries]
                household_ids_per_region_corrected = []
                for household_ids in household_ids_per_region:
                    if household_ids.size > 0:
                        household_ids_per_region_corrected.append(household_ids + c)
                        c += household_ids[-1] + 1
                merged_move_dictionary[key] = np.hstack(household_ids_per_region_corrected)
                self.model.logger.info(f"Moving {len(merged_move_dictionary[key])} agents")
            else:
                merged_move_dictionary[key] = np.hstack([d[key] for d in move_dictionaries])
        return merged_move_dictionary

    def step(self):
        for households in self.all_households:
            households.step()
            
        move_dictionaries = []
        for i, region in enumerate(self.all_households):
            move_dictionary = region.move()
            self.model.logger.info(f"({i+1}/{len(self.all_households)}) - Moving {len(move_dictionary['household_id']) if move_dictionary else 0} households from {region.geom_id}")


            if move_dictionary:
                move_dictionaries.append(move_dictionary)
        if move_dictionaries:
            # merge movement data of different regions in 1 dictionary
            merged_move_dictionary = self.merge_move_dictionary(move_dictionaries)
            if self.model.settings['general']['export_move_dictionary']:
                merged_pd = pd.DataFrame(merged_move_dictionary)#.to_csv(os.path.join(self.model.config['general']['report_folder'], f'move_dictionary_{self.model.current_time.year}.csv'))
                
                unique_household_ids = merged_pd.groupby('household_id').mean().drop(columns = ['gender', 'age'])     

                fn = os.path.join(self.model.config['general']['report_folder'], 'move_dictionaries')
                if not os.path.exists(fn):
                    os.makedirs(fn)
                
                from_region = [self.model.agents.regions.ids[int(i)] for i in unique_household_ids['from']]
                to_region = [self.model.agents.regions.ids[int(i)] for i in unique_household_ids['to']]
                unique_household_ids['from'] = from_region
                unique_household_ids['to'] = to_region
                coastal = [coast_key for coast_key in list(unique_household_ids['from']) if coast_key.endswith('flood_plain')]
                unique_household_ids_filt = unique_household_ids[unique_household_ids['from'].isin(coastal)]
                unique_household_ids_filt.to_csv(os.path.join(fn, f'move_dictionary_housholds_{self.model.current_time.year}.csv'))

            # split the movement data by destination region and send data to region

            sort_idx = np.argsort(merged_move_dictionary['to'], kind='stable')
            for key, value in merged_move_dictionary.items():
                merged_move_dictionary[key] = value[sort_idx]
            move_to_regions, start_indices = np.unique(merged_move_dictionary['to'], return_index=True)
            end_indices = np.append(start_indices[1:], merged_move_dictionary['to'].size)
       
            for region_idx, start_idx, end_idx in zip(move_to_regions, start_indices, end_indices):
                move_data_per_region = {
                    key: value[start_idx: end_idx]
                    for key, value in merged_move_dictionary.items()
                }

                self.all_households[region_idx].add(move_data_per_region)   
       
        # Export all agents to csv
        if self.model.settings['general']['export_agents']:
            export_agents(
            agent_locs = [household.locations for household in self.agents.regions.all_households],
            agent_size = self.agents.regions.household_sizes,
            agent_adapt = self.agents.regions.household_adapted,
            agent_income = self.agents.regions.household_incomes,
            agent_risk_perception = self.agents.regions.household_risk_perception, 
            agent_ead = self.agents.regions.household_ead,
            agent_since_flood = self.agents.regions.since_flood,
            year = self.model.current_time.year, 
            export = True,
            report_folder = self.model.config['general']['report_folder']
            )   

        if self.model.settings['general']['export_matrix'] and move_dictionaries:
            # Export matrix to csv
            export_folder = os.path.join(self.model.config['general']['report_folder'], 'migration_matrices')
            export_matrix(geoms = self.model.agents.regions.geoms, 
                        dest_folder=export_folder, 
                        move_dictionary=merged_move_dictionary,
                        year=self.model.current_time.year)