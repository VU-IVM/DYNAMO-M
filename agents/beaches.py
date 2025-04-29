from honeybees.agents import AgentBaseClass
import numpy as np
from honeybees.library.raster import pixel_to_coord, coords_to_pixels
import rasterio
import os
from affine import Affine


class Beaches(AgentBaseClass):
    '''This class contains all beaches simluted in the model.'''
    
    def __init__(self, model, agents):
        self.model = model
        self.agents = agents
        self.ids = self.agents.regions.ids
        AgentBaseClass.__init__(self)

        # load initial beach width
        self.initial_beach_width = self.model.settings['shoreline_change']['initial_beach_width']
        sandy_beach_cells = model.data.sandy_beach_cells.get_data_array().astype(np.int16)
        self.indices_beach_cells = np.where(sandy_beach_cells == 1)
        self.beach_width = np.full(sandy_beach_cells.shape, -1, np.float32)
        self.beach_width[self.indices_beach_cells] = self.initial_beach_width
        self.beach_data_gt = self.model.data.sandy_beach_cells.gt
        del sandy_beach_cells # no longer needed

        # get depth of closure
        self.beach_depth_of_closure = model.data.depth_of_closure.get_data_array()[self.indices_beach_cells]

        # initiate perc beach lost
        self.percentage_beach_lost = 0
        self.volume_sand_eroded = 0

        self._initiate_admin_indices_beach_cells()
        self._initiate_shoreline_change()

    def _initiate_admin_indices_beach_cells(self):
        self.admin_indices_beach_cells = {}
        for household in self.agents.regions.all_households:
            if household.geom_id.endswith('flood_plain') and self.model.config['general']['create_agents']:
                self.admin_indices_beach_cells[household.geom_id] = coords_to_pixels(coords = household.locations_admin_cells, gt = self.beach_data_gt)     

    def _initiate_shoreline_change(self):
        'This function fit an exponential on the datapoints in the 2050 and 2100 shoreline projections to capture the effect of increasing sea level rise rates'
        # get change rates
        self.shoreline_change_trend = self.model.data.shoreline_change_trend.get_data_array()[self.indices_beach_cells]
        
        if self.model.args.rcp == 'control': 
            self.polynomials_slr_erosion = np.full((3, self.indices_beach_cells[0].size), 0 , np.float32)
        else: 
            self.shoreline_loss_2050_slr = self.model.data.shoreline_loss_2050.get_data_array()[self.indices_beach_cells]
            self.shoreline_loss_2100_slr = self.model.data.shoreline_loss_2100.get_data_array()[self.indices_beach_cells]

            shoreline_lost_2010 = np.full(self.indices_beach_cells[0].size, 0, np.float32)
            shoreline_lost_2050 = self.shoreline_loss_2050_slr
            shoreline_lost_2100 = self.shoreline_loss_2100_slr

            # define x and y for fitting
            x = np.array([2010, 2050, 2100])
            y = np.array([shoreline_lost_2010, shoreline_lost_2050, shoreline_lost_2100])
            
            # estimate polynomial of 2 degrees to each beach cell to allow for exponential increases in slr induced erosion
            self.polynomials_slr_erosion = np.polyfit(x, y, deg=2)


    def interpolate_slr_erosion(self):
        # apply to estimate annual erosion rate under SLR       
        shoreline_position_past_year =  self.polynomials_slr_erosion[0, :] * (self.model.current_time.year-1) ** 2 + self.polynomials_slr_erosion[1, :] * (self.model.current_time.year-1) + self.polynomials_slr_erosion[2, :]
        shoreline_position_current_year =  self.polynomials_slr_erosion[0, :] * (self.model.current_time.year) ** 2 + self.polynomials_slr_erosion[1, :] * (self.model.current_time.year) + self.polynomials_slr_erosion[2, :]
        self.slr_induced_erosion = shoreline_position_current_year - shoreline_position_past_year


    def process_shoreline_change(self):
        if self.model.settings['shoreline_change']['include_erosion'] and not self.model.spin_up_flag:
            self.interpolate_slr_erosion()        
            
            self.beach_width[self.indices_beach_cells] += (self.shoreline_change_trend + self.slr_induced_erosion)
            # filter negative beach width
            self.beach_width[self.indices_beach_cells] = np.maximum(
                self.beach_width[self.indices_beach_cells], 0)
            # do not allow for beaches wider than initial width
            self.beach_width[self.indices_beach_cells] = np.minimum(
                self.beach_width[self.indices_beach_cells], self.initial_beach_width)

            self.percentage_beach_lost = round(np.sum(self.beach_width[self.indices_beach_cells] == 0) / self.beach_width[self.indices_beach_cells].size * 100)

            # calculate total volume of sand eroded
            indices_to_fill = np.where(np.logical_and(self.beach_width != -1, self.beach_width < self.initial_beach_width))
            area_to_fill = (self.initial_beach_width - self.beach_width[indices_to_fill]).astype(np.float64) * 1E6

            # get indices to fill relative to beach indices
            indices_to_fill_rel = np.where(self.beach_width[self.indices_beach_cells] < self.initial_beach_width)

            volumes_to_fill = area_to_fill * \
                self.beach_depth_of_closure[indices_to_fill_rel]
            self.volume_sand_eroded = np.sum(volumes_to_fill)            


    def export_beach_width_to_tiff(self):
        with rasterio.open(os.path.join('DataDrive', 'EROSION', 'PROCESSED', 'rasterized_changerate_world_rcp4p5_adjusted.tif')) as src:
            out_meta = src.meta

        out_meta['transform'] = Affine.from_gdal(*self.beach_data_gt)
        out_meta['width'] = self.beach_width.shape[1]
        out_meta['height'] = self.beach_width.shape[0]
        out_meta['nodata'] = -1

        target_tiff = f'report/{self.model.args.beach_manager_strategy}_beach_width_{self.model.args.rcp}_{self.model.current_time.year}.tif'
        with rasterio.open(target_tiff, "w", **out_meta) as dest:
            dest.write(self.beach_width, indexes=1)

    def step(
            self):
        self.process_shoreline_change()
        # if self.model.current_time.year in [2015, 2020, 2025, 2030, 2035]:
            # self.export_beach_width_to_tiff()
            # self.export_times_renourished_to_tiff()
            # self.export_beach_renourishment_costs_to_tiff()