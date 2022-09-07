'''Functions used to draw the agents and nodes when the model is run with model visualization.'''

from honeybees.artists import Artists
import random
import math
import numpy as np
from scipy import stats
import numpy as np 
import matplotlib


class ArtistsCOASTMOVE(Artists):
    def __init__(self, model):
        Artists.__init__(self, model)

    def draw_regions_household(self, model, agents, idx, area, *args, **kwargs):
        household_size = agents.size[idx].item()
        # r = 0 if household_size == 0 else math.log(household_size) # Visualize width of the dot based on household size
        r = 0 if household_size == 0 else math.log(3) # Visualize width of the dot based on household size

        # breakpoint()
        adapted = agents.adapt[idx]
        
        # initiate color ramp for risk perceptions
        cmap = matplotlib.cm.get_cmap('Reds')

        flooded = agents.flooded[idx]
        risk_perception = agents.risk_perception[idx]
        if flooded == 1:
            color = 'blue'
        elif adapted == 1:
            color = 'green'
        elif risk_perception > 0.5:
            color = cmap((risk_perception/5*0.8))
            color = matplotlib.colors.rgb2hex(color)
        else:
            color = 'grey'
        
        # household_indices = agents.people_indices_per_household[idx, :household_size]
        # gender = agents.gender[household_indices]
        # if stats.mode(gender).mode.item() == 0:
        #     breakpoint()
        #     color = "green"
        # else:
        #     color = "red"
        return {"type": "shape", "shape": "circle", "r": r, "filled": True, "color": color}

    def draw_regions_aggregate_household(self, model, agent, *args, **kwargs):
        r = 0 if agent.n == 0 else np.log(agent.n+100)
        # r = 0 if agent.n == 0 else 5
        return {"type": "shape", "shape": "circle", "r": r, "filled": True, "color": 'orange'}

    def draw_country(self, color, filled):
        return {"type": "shape", "shape": "polygon", "filled": filled, "color": color, "edge": True}  # start without fill or edge, later colored through update

    def draw_admin(self, color, filled, lower=None, upper=None):
        return {"type": "shape", "shape": "polygon", "filled": filled, "color": color, "edge": True}  # start without fill or edge, later colored through update

    def update_admin(self, ID, portrayal, *args, **kwargs):
        # minimum = 0
        # maximum = 10
        # v = min(255, max(0, int((self.model.agents.admin.gdp[ID] - minimum) / (maximum - minimum) * 255)))
        # portrayal['color'] = '#0000FF%02X' % (v, )
        pass
        # if 'flood_plane' in ID:
        #     portrayal['color'] = 'black'
        # else:
        #     portrayal['color'] = 'brown'             