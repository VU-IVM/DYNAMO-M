import scipy
import numpy as np


class CoastalAmenities:
    def __init__(self, model) -> None:
        self.model = model
        amenity_functions = self.model.data.coastal_amenity_functions

        self.lims_beach_width = min(
            amenity_functions['beach_amenity'].index), max(
            amenity_functions['beach_amenity'].index)
        self.lims_dist2coast = min(
            amenity_functions['dist2coast'].index), max(
            amenity_functions['dist2coast'].index)

        self.beach_amenity_function = scipy.interpolate.interp1d(
            x=amenity_functions['beach_amenity'].index,
            y=amenity_functions['beach_amenity']['premium'])

        self.coastal_function = scipy.interpolate.interp1d(
            x=amenity_functions['dist2coast'].index,
            y=amenity_functions['dist2coast']['premium'])

    def calculate_beach_amenity(
        self,
        agent_wealth,
        beach_width: np.ndarray,
        beach_proximity_bool: bool,
    ) -> np.ndarray:
        '''Function used to calculate the amenity value of a beach.
        Args:
            coastal_amenity_function (scipy.interpolate.interpolate.interp1d): function used to calculate the amenity value of a beach
            beach_width (np.ndarray): beach width in floodplain.
            beach_proximity_bool (bool): boolean indicating whether the agent is located adjecent to a beach or not.
            max_beach_amenity_value (int): maximum amenity value of a beach.
        Returns:
            beach_amenity: amenity value of a beach per agent in euros.

        '''

        # just a quick linear function to interpolate amenity value based on
        # shoreline retreat
        assert (beach_proximity_bool != -1).all()

        beach_amenity_function = self.beach_amenity_function

        # Filter values outside interpolation range
        beach_width_adj = np.maximum(
            beach_width, self.lims_beach_width[0])
        beach_width_adj = np.minimum(
            beach_width_adj, self.lims_beach_width[1])

        # Set values to zero (default)
        beach_amenity = np.zeros(agent_wealth.size, dtype=np.float32)

        # Calculate
        calculated_beach_amenity = beach_amenity_function(
            beach_width_adj[beach_proximity_bool]) * agent_wealth[beach_proximity_bool]
        beach_amenity[beach_proximity_bool] = calculated_beach_amenity

        return beach_amenity

    def calculate_coastal_amenity(
        self,
        distance_to_coast: np.ndarray,
        agent_wealth: np.ndarray
    ) -> np.ndarray:
        '''This function calculates the coastal amenity value experienced by the agent based on distance to coast, erosion rate and wealth.

        Args:
            distance_to_coast (np.ndarray): distance to coast in meters
            shoreline_change (np.ndarray): shoreline change in meters per year
            agent_wealth (np.ndarray): agent wealth in euros

        Returns:
            np.ndarray: amenity value in euros
        '''
        # Initiate amenity distance decay function (interpolate between values of
        # Conroy & Milosch 2011)

        coastal_function = self.coastal_function

        # Filter values outside interpolation range
        distance_to_coast = np.maximum(
            distance_to_coast, self.lims_dist2coast[0])
        distance_to_coast = np.minimum(
            distance_to_coast, self.lims_dist2coast[1])

        return coastal_function(distance_to_coast)

    def total_amenity(
            self,
            beach_proximity_bool,
            distance_to_coast,
            beach_width,
            agent_wealth,
    ):

        coastal_amenity_premium = self.calculate_coastal_amenity(
            distance_to_coast=distance_to_coast,
            agent_wealth=agent_wealth
        )

        if beach_proximity_bool.sum() > 0:
            beach_amenity = self.calculate_beach_amenity(
                agent_wealth=agent_wealth,
                beach_width=beach_width,
                beach_proximity_bool=beach_proximity_bool,
            )

        else:
            beach_amenity = np.zeros_like(agent_wealth)
        
        coastal_amenity = coastal_amenity_premium * agent_wealth
        # return amenity value
        # amenity_value = coastal_amenity + beach_amenity
        amenity_value = coastal_amenity
        return coastal_amenity_premium, amenity_value, beach_amenity

    def update_amenity_functions(self):
        amenity_functions = self.model.data.coastal_amenity_functions

        self.lims_beach_width = min(
            amenity_functions['beach_amenity'].index), max(
            amenity_functions['beach_amenity'].index)
        self.lims_dist2coast = min(
            amenity_functions['dist2coast'].index), max(
            amenity_functions['dist2coast'].index)

        self.beach_amenity_function = scipy.interpolate.interp1d(
            x=amenity_functions['beach_amenity'].index,
            y=amenity_functions['beach_amenity']['premium'])

        self.coastal_function = scipy.interpolate.interp1d(
            x=amenity_functions['dist2coast'].index,
            y=amenity_functions['dist2coast']['premium'])