import numpy as np
import os
import pandas as pd
import geopandas as gpd
from honeybees.agents import AgentBaseClass
from scipy.spatial import distance_matrix as sdistance_matrix
from agents.coastal_nodes import CoastalNode
from agents.inland_nodes import InlandNode
from agents.node_properties import NodeProperties
import geopandas as gpd
from utils.mapping import create_agent_geotif

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
        self._amenity_mapping()

    def initiate_agents(self):
        self.geoms = self.model.area.geoms['admin']       
        self.n = len(self.geoms)
        self._initiate_attributes()

        max_household_size = 15

        self.household = {}
        self.aggregate_household = {}
        self.all_households = []
        j = 0
        n_households_per_region = np.full(len(self.geoms), 0, dtype=np.int32)      
        i = 0
        for geom in self.geoms:

            # All nodes
            ID = geom['properties']['id']
            # Coastal nodes
            if ID.endswith(
                    'flood_plain') & self.model.config['general']['create_agents']:
                if self.model.args.subdivision == 'GADM':
                    init_folder = os.path.join(
                        "DataDrive",
                        "SLR",
                        f"households_gadm_{self.model.args.admin_level}_{self.model.config['general']['start_time'].year}",
                        self.model.config['general']['size'],
                        ID.replace(
                            '_flood_plain',
                            ''))
                elif self.model.args.subdivision == 'GDL':
                    init_folder = os.path.join(
                        "DataDrive",
                        "SLR",
                        f"households_gdl_{self.model.config['general']['start_time'].year}",
                        self.model.config['general']['size'],
                        ID.replace(
                            '_flood_plain',
                            '')) 
                
                locations_fn = os.path.join(init_folder, "locations.npy")
                # assert if locations exist (otherwise bug in find study area)
                if not os.path.exists(locations_fn):
                    raise FileNotFoundError(f'Agents not found for {ID}. Check start year of model run (2015)')
                # redundancy of 200%, min of 500_000
                redundancy = max(int(np.load(locations_fn).size*3), 1_000_000)
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
                i+= 1
            else:
                # replace floodplain with coastal when runnning without agents
                geom['properties']['id'] = geom['properties']['id'].replace('flood_plain', 'coastal')
                household_class = InlandNode(
                    self.model,
                    self.agents,
                    idx=i,
                    geom=geom,
                    distance_matrix=self.distance_matrix,
                    n_households_per_region=n_households_per_region,
                    max_household_size=max_household_size
                )
                i += 1
                j += 1
                self.aggregate_household[ID] = household_class
            self.all_households.append(household_class)
            print(f'Building {i}/ {self.n} ({household_class.geom_id})')

        self.model.logger.info(
            f'Created {sum(household.n for household in self.household.values())} households')
        self.model.logger.info(
            f'Created {sum(household.n for household in self.aggregate_household.values())} aggregrate households')
        
    
    def _load_initial_state(self):
        return None

    def _get_distance_matrix(self):
        centroids = self.centroids

        # Convert (back) to geopandas points
        gpd_points = gpd.points_from_xy(
            centroids[:, 0], centroids[:, 1], crs='EPSG:4326')

        # Project points in world Mollweide (?)
        gpd_points = gpd_points.to_crs('ESRI:54009')

        # Extract x and y
        x = gpd_points.x
        y = gpd_points.y

        # Stack into dataframe and export distance matrix
        projected_centroids = np.column_stack((x, y))
        return sdistance_matrix(
            projected_centroids,
            projected_centroids) / 1000  # Devide by 1000 to derive km

    def _initiate_attributes(self):
        self.distance_matrix = self._get_distance_matrix()

    def _amenity_mapping(self): 
        # dict to store the indices of the premiums
        amenity_premium_indices = {}
        
        # get countries in model
        countries_in_model = np.unique([geom['properties']['id'][:3] for geom in self.geoms])
        for iso3 in countries_in_model:
            # iterate over geoms in country
            # get coastal nodes in country
            coastal_nodes_in_country = [household for household in self.all_households if household.geom_id.startswith(iso3) and isinstance(household, CoastalNode)]
            # only continue if there are more than 1 coastal nodes
            if len(coastal_nodes_in_country) > 1:
                # collect all unique coastal amenity premiums in country
                all_amenity_premiums = np.unique(np.concatenate([coastal_node.coastal_amenity_premium_cells for coastal_node in coastal_nodes_in_country]))
                # all_amenity_premiums = np.arange(0, 1, 0.02, np.float32)
                # iterate over all unique premiums
                # find indice of premium in each coastal node and store in dictionary
                for coastal_node in coastal_nodes_in_country:
                    amenity_premium_indices[coastal_node.admin_idx] = {}
                    for premium in all_amenity_premiums:
                        indice_premiums = np.where(coastal_node.coastal_amenity_premium_cells <= premium)[0]
                        indice_premiums = np.where(coastal_node.coastal_amenity_premium_cells <= premium)[0]
                        amenity_premium_indices[coastal_node.admin_idx][premium] = indice_premiums
                # add results to dictionary
        # store
        self.amenity_premium_indices = amenity_premium_indices


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
                        household_ids_per_region_corrected.append(
                            household_ids + c)
                        c += household_ids[-1] + 1
                merged_move_dictionary[key] = np.hstack(
                    household_ids_per_region_corrected)
                self.model.logger.info(
                    f"Moving {len(merged_move_dictionary[key])} agents")
            else:
                merged_move_dictionary[key] = np.hstack(
                    [d[key] for d in move_dictionaries])
        return merged_move_dictionary

    def parse_move_dictionaries(self, dictionary_to_parse, export=False):
        move_data_per_regions = []
        
        merged_move_dictionary = self.merge_move_dictionary(
            dictionary_to_parse)

        if export:
            self.process_migration_matrices(merged_move_dictionary)
        
        # split the movement data by destination region and send data to
        # the region
        sort_idx = np.argsort(merged_move_dictionary['to'], kind='stable')
        for key, value in merged_move_dictionary.items():
            merged_move_dictionary[key] = value[sort_idx]
        move_to_regions, start_indices = np.unique(
            merged_move_dictionary['to'], return_index=True)
        end_indices = np.append(
            start_indices[1:], merged_move_dictionary['to'].size)

        for region_idx, start_idx, end_idx in zip(
                move_to_regions, start_indices, end_indices):
            move_data_per_region = {
                key: value[start_idx: end_idx]
                for key, value in merged_move_dictionary.items()
            }

            # check if migration does not exceed limits set in settings
            if self.model.settings['decisions']['migration']['limit_admin_growth'] and self.all_households[region_idx].geom_id.endswith('flood_plain'): 
                n_households_in_region = self.all_households[region_idx].n
                # check if this is a person or household move dictionary
                
                if 'household_id' in move_data_per_region.keys():
                    # if person move dictionary get n households from unique ids
                    n_households_moving_in = np.unique(move_data_per_region['household_id']).size
                else:
                    # if household move dictionary get n households from size of from array
                    n_households_moving_in = move_data_per_region['from'].size
                
                # at least always allow for household from gravity model to move in (these will be removed by gravity model anyway)
                max_growth_allowed = (2 * self.model.settings['decisions']['migration']['max_admin_growth'] * n_households_in_region)
                max_growth_allowed = max(max_growth_allowed, np.unique(move_data_per_region['from']).size)
                if n_households_moving_in > max_growth_allowed: # a factor 2 to account for some buffer
                    raise ValueError(f"Migration to {self.all_households[region_idx].geom_id} exceeds limit of {self.model.settings['decisions']['migration']['max_admin_growth']} households")

            move_data_per_regions.append(move_data_per_region)
        
        return move_data_per_regions, move_to_regions

    def export_agent_exposure(self):
        # get data array to write to
        gt = self.model.data.population.gt
        empty_array = self.model.data.population.get_data_array() * 0             
        
        # create folder to store
        folder_to_store_tifs = os.path.join(self.model.reporter.export_folder, 'agent_tiffs')
        os.makedirs(folder_to_store_tifs, exist_ok=True)

        # create filename
        if self.model.spin_up_flag:
            fn = os.path.join(folder_to_store_tifs, f'exposure_spin_up_{self.model.spin_up_cycle}.tif')
        else:
            fn = os.path.join(folder_to_store_tifs, f'exposure_{self.model.current_time.year}.tif')

        # get all locations and sizes
        locations = [household.locations for household in self.all_households if isinstance(household, CoastalNode)]
        locations = np.concatenate(locations)
        property_values = [household.property_value for household in self.all_households if isinstance(household, CoastalNode)]
        property_values = np.concatenate(property_values)

        # write array to geotiff
        create_agent_geotif(
            array = empty_array,
            attribute = property_values,
            coords = locations,
            gt = gt,
            output_fn = fn
        )


    def export_agent_density(self):
        # get data array to write to
        gt = self.model.data.population.gt
        empty_array = self.model.data.population.get_data_array() * 0             
        
        # create folder to store
        folder_to_store_tifs = os.path.join(self.model.reporter.export_folder, 'agent_array')
        os.makedirs(folder_to_store_tifs, exist_ok=True)

        # create filename
        if self.model.spin_up_flag:
            fn = os.path.join(folder_to_store_tifs, f'population_spin_up_{self.model.spin_up_cycle}.tif')
        else:
            fn = os.path.join(folder_to_store_tifs, f'population_{self.model.current_time.year}.tif')

        # get all locations and sizes
        locations = [household.locations for household in self.all_households if isinstance(household, CoastalNode)]
        locations = np.concatenate(locations)
        sizes = [household.size for household in self.all_households if isinstance(household, CoastalNode)]
        sizes = np.concatenate(sizes)
        fill = np.full(sizes.size, 1, np.int32)

        # write array to geotiff
        create_agent_geotif(
            array = empty_array,
            attribute = fill,
            coords = locations,
            gt = gt,
            output_fn = fn
        )

    def export_coastal_fps_dictionaries(self):
        merged_dict = self.all_coastal_fps_gov
        export_folder = os.path.join(self.model.reporter.export_folder, 'coastal_fps')

        if not hasattr(self, 'merged_pd'):
            self.merged_pd = pd.DataFrame(merged_dict, index = [f'{self.model.current_time.year}']).transpose()
            os.makedirs(export_folder, exist_ok=True)
            self.merged_pd.to_csv(os.path.join(export_folder, f'coastal_fps.csv'))
        else:
            os.makedirs(export_folder, exist_ok=True)
            fps_pd = pd.DataFrame(merged_dict, index = [f'{self.model.current_time.year}']).transpose()
            self.merged_pd = pd.concat([self.merged_pd, fps_pd], axis = 1)
            self.merged_pd.to_csv(os.path.join(export_folder, f'coastal_fps.csv'))

    def reset_node_attributes(self,
        attributes=[
                    # 'fraction_intending_to_migrate',
                    # 'n_intending_to_migrate',
                    'n_households_that_would_have_moved',
                    'n_people_that_would_have_moved',
                    'n_households_that_would_have_adapted', 
                    'n_people_that_would_have_adapted',
                    'household_spendings',
                    'household_spendings_relative_to_gdp',
                    'n_persons_moving_out',
                    'n_persons_moving_in',
                    'n_adapted_movers',
                    'fraction_adapted_movers']):
        
        for household in self.all_households:
            for atrribute in attributes:
                if hasattr(household, atrribute):
                    setattr(household, atrribute, getattr(household, atrribute) * 0)
            

    def step(self):
        # only select nodes with population
        self.model.logger.info(f'Current simulation year: {self.model.current_time.year}')
        self.reset_node_attributes()
        person_move_dictionaries = []
        household_move_dictionaries = []

        # create snapshot of population in beginning of step for use in gravity and migration decisions
        self.population_snapshot = tuple(self.population)

        # create snapshot of ead per region
        ead_nodes = [ead if ead != None else 0 for ead in self.ead_nodes]
        
        # quick fix for empty regions
        agents_in_simulation = np.maximum(self.agents_in_simulation, 1)
        self.average_ead_snapshot =tuple(np.array(ead_nodes)/ agents_in_simulation)

        # get snapshot of amenity premium and damage factors
        self.snapshot_damages_cells = tuple([household.damages_coastal_cells if hasattr(household, 'damages_coastal_cells') else np.array([0]) for household in self.all_households])
        self.snapshot_amenity_premium_cells = tuple([household.coastal_amenity_premium_cells if hasattr(household, 'coastal_amenity_premium_cells') else np.array([0]) for household in self.all_households])
        # only consider nodes with a population for move and add
        all_nodes_with_population = [
            node for node in self.all_households if node.n > 0]

        for i, node in enumerate(all_nodes_with_population):
            if self.model.args.low_memory_mode:
                node.load_arrays_from_npz()
            
            # government step for node
            self.model.agents.government.step(node)
            
            # insurer agent step (only for coastal nodes)
            if node.geom_id.endswith('flood_plain'):
                self.model.agents.insurer.step(node)

            # Update risk information and process population change
            node.step()
            
            # execute the move function (both gravity or ABM depending on node type)
            person_move_dictionary, household_move_dictionary = node.move()
            self.model.logger.info(
                f"({i+1}/{len(all_nodes_with_population)}) - Moving {len(person_move_dictionary['household_id']) if person_move_dictionary else 0} households from {node.geom_id}")
           
            if person_move_dictionary:
                person_move_dictionaries.append(person_move_dictionary)
                household_move_dictionaries.append(household_move_dictionary)
                node.n_persons_moving_out = len(person_move_dictionary['household_id'])
            if self.model.args.low_memory_mode:
                node.save_arrays_to_npz()

        # first process household level attributes
        if person_move_dictionaries:
            # merge movement data of different regions in 1 dictionary           
            export = self.model.settings['general']['export_matrix']
            person_move_data_per_regions, move_to_regions = self.parse_move_dictionaries(person_move_dictionaries, export=export)
            household_move_data_per_regions, _ = self.parse_move_dictionaries(household_move_dictionaries)
            
            # now allocate move dictionaries to regions
            for i, region_idx in enumerate(move_to_regions):
                if self.model.args.low_memory_mode:
                    self.all_households[region_idx].load_arrays_from_npz()
                self.all_households[region_idx].n_persons_moving_in = len(person_move_data_per_regions[i]['household_id'])
                self.all_households[region_idx].add(person_move_data_per_regions[i], household_move_data_per_regions[i])

                if self.model.args.low_memory_mode:
                    self.all_households[region_idx].save_arrays_to_npz()
        
        if self.model.settings['general']['export_agent_tiffs']:
            if self.model.current_time.year in [2016, 2020, 2040, 2060, 2080]:
                self.export_agent_density()
                # self.export_agent_exposure()
            
        self.export_coastal_fps_dictionaries()

    def process_migration_matrices(self, merged_move_dictionary):
        geom_ids = np.array([geom['properties']['id'] for geom in self.geoms])
        start = 0
        # create empy pd df
        matice_pd = pd.DataFrame()
        origins, counts = np.unique(merged_move_dictionary['from'], return_counts=True)
        for origin, count in zip(origins, counts):
            # get n moving out per destination
            dest, flow  = np.unique(merged_move_dictionary['to'][start:start+count], return_counts=True)
            # get IDs in model 
            to_add = pd.DataFrame({
                'origin': np.full(dest.size, np.take(geom_ids, origin)), 
                'destination': np.take(geom_ids, dest),
                'flow': flow
                })
            assert origin not in dest
            matice_pd = pd.concat([matice_pd, to_add])
            start += count
        if True:
            folder_path = os.path.join(self.model.reporter.export_folder, 'migration_matrices')
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            if self.model.spin_up_flag:
                matice_pd.to_csv(os.path.join(folder_path, f'migration_matrix_spin_{self.model.spin_up_cycle}.csv'), index=False)
            else:
                matice_pd.to_csv(os.path.join(folder_path, f'migration_matrix_{self.model.current_time.year}.csv'), index=False)

    def collect_input_data(self):
        # iterate over all households and export input data for inspection
        attrs_to_collect = ['income_region', 'initial_fps', 'adaptation_cost', 'dike_elevation_cost', 'fixed_migration_cost']
        collected_attributes = {}
        for household in self.all_households:
            collected_attributes[household.geom_id] = {}
            for attr in attrs_to_collect:
                if hasattr(household, attr):
                    collected_attributes[household.geom_id][attr] = getattr(household, attr)
        # save
        if isinstance(self.model.args.area, list):
            area = self.model.args.area[0]
        else: area = self.model.args.area
        report_folder = self.model.config['general']['report_folder']
        os.makedirs(report_folder, exist_ok = True)
        pd.DataFrame(collected_attributes).transpose().to_csv(os.path.join(report_folder, f'input_values_{area}.csv'))
