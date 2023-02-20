from honeybees.agents import AgentBaseClass
import numpy as np

class Government(AgentBaseClass):
    def __init__(self, model, agents):
        self.model = model
        self.agents = agents
        AgentBaseClass.__init__(self)

        # Retrieve the ead in the node

    def hold_the_line(
        self,
        beach_slope = 1.5,
        desired_width = 100,
        minimum_width = 80,
        m3_per_m = 200, # Lit at 200 for NL
        eur_m_coast = 2_000,
        sand_surcharge = 1.2, # Additional sand required to offset other processes
        cost_m3_sand = 6,
        fixed_cost = 100_000,

        ):
    
        slope_rad = np.deg2rad(beach_slope) 

        # Iterate over all beaches. If average beach lenght is less than 80m refill eroding segment to 100m (should maybe vectorize later)
        beach_dict = self.model.agents.beaches.beach_dict
        for beach_id in beach_dict.keys():
            if beach_dict[beach_id]['average_width_m'] < minimum_width:
                self.model.agents.beaches.beach_width[beach_dict[beach_id]['segments']] = np.maximum(self.model.agents.beaches.beach_width[beach_dict[beach_id]['segments']], desired_width)
                    
                                     
   
    def national(self):
        # Methods that act on the national/ global scale
        self.hold_the_line()

    
    def regional(self):
        # Methods that act on the subnational/ admin scale
        pass

    def step(self):
        self.national()
        self.regional()
        # Retrieve the loss in beach amenity value in the node

        # Do something with NPVs and cost benefit analysis to raise FPS 
           # Do something with NPVs and cost benefit analysis to renourish beaches