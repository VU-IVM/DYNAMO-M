from honeybees.agents import AgentBaseClass
import numpy as np

class Beaches(AgentBaseClass):
    
    def __init__(self, model, agents):
        self.model = model
        self.agents = agents
        AgentBaseClass.__init__(self)

        self.beach_ids = self.model.data.beach_ids
        self.ids = np.unique(np.array(self.model.data.beach_ids['beach_ID_1']))
        self.shoreline_projections = self.model.data.shoreline_projections
        self.beach_width = np.full(self.shoreline_projections.shape[0], 100, dtype=np.float32)
        self.segment_locations = np.stack([self.shoreline_projections[:, 1], self.shoreline_projections[:, 0]], axis = 1)
        self.value_segment = np.full(self.shoreline_projections.shape[0], 0, dtype=np.float32)
        self.store_beach_attributes()


    def interpolate_shoreline_change(
        self
        ) -> np.ndarray:

        '''This function interpolates the shoreline change rate for a given year.
        Args:
            polynomials: array containing the coefficients of the polynomials for each coastal segment
            year: year for which to interpolate the shoreline change rate
        Returns:
            shoreline_change: interpolated shoreline change rate for each coastal segment
        '''
        polynomials = self.shoreline_projections[:, 2:]
        current_year= self.model.current_time.year

        change_current_year = polynomials[:, 0] * current_year ** 2 + polynomials[:, 1] * current_year + polynomials[:, 2]
        change_previous_year = polynomials[:, 0] * (current_year-1) ** 2 + polynomials[:, 1] * (current_year-1) + polynomials[:, 2]
        return change_current_year - change_previous_year
    
    def process_shoreline_change(
            self
            ):
        shoreline_change = self.interpolate_shoreline_change()
        self.beach_width  = np.maximum(0, self.beach_width + shoreline_change)

    def store_beach_attributes(
        self
        ):
        # This is very slow. Also only required when renourishment decisions are made and for export. 
        beach_dict = {}

        for beach_id in self.beach_ids['beach_ID_1']:
            beach_dict[beach_id] = {}
            segments = np.array(self.beach_ids[self.beach_ids['beach_ID_1'] == beach_id].index)
            beach_dict[beach_id]['segments'] = segments
            beach_dict[beach_id]['value_beach_EUR'] = np.sum(self.value_segment[segments])
            beach_dict[beach_id]['average_width_m'] = np.mean(self.beach_width[segments])
            beach_dict[beach_id]['beach_length_m'] = segments.size * 250
        
        self.beach_dict = beach_dict
        
    def step(
        self
        ): 
        if self.model.settings['shoreline_change']['include_erosion']:
            self.process_shoreline_change()
            self.store_beach_attributes()
