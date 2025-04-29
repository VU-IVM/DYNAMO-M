import geopandas as gpd
import pandas as pd
import os
import numpy as np
import warnings
from create_government_geoms import rasterize_geoms

# Suppress specific UserWarning
warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.*")



def load_geoms(path):
    """Load geoms from path."""
    return gpd.read_file(path)

def load_coastline(path):
    """Load coastline from path."""
    coastline = gpd.read_file(path)
    coastline = coastline.to_crs('EPSG:4326')
    return coastline

def iterate_over_geoms(geoms, coastline):
    
    # assign attribute to geom to indicate it has a floodplain
    geoms['coastal'] = False
    geoms['merged'] = False
    
    # iterate over geoms and set attribute to true if it has a floodplain
    for i, geom in geoms.iterrows():
        if geom.geometry.intersects(coastline.geometry)[0]:
            geoms.at[i, 'coastal'] = True
    
       
    # now iterate over geoms that do not have a floodplain
    iter = 0
    while not geoms[geoms['coastal']==False].empty:
        print(f'iter={iter}')
        # geoms_to_remove = []
        # geoms_to_add = gpd.GeoDataFrame(columns=geoms.columns, crs=geoms.crs)
        
        for i, geom in geoms[geoms['coastal']==False].iterrows():
            neighbors = gpd.GeoDataFrame(columns=geoms.columns, crs=geoms.crs)

            # Check if the geom is adjacent to a geom that has a floodplain
            for j, coastal_geom in geoms[geoms['coastal'] == True].iterrows():
                if geom.geometry.touches(coastal_geom.geometry):
                    # Wrap the coastal_geom Series in a DataFrame before concatenation
                    coastal_geom_df = gpd.GeoDataFrame([coastal_geom], columns=geoms.columns, crs=geoms.crs)
                    neighbors = pd.concat([neighbors, coastal_geom_df], ignore_index=True)
                
            # get closest neighbor
            if not neighbors.empty:
                # get closest neighbor
                closest_neighbor = neighbors.iloc[neighbors.distance(geom.geometry).idxmin()]
                
                # some checks
                name_closest_geom = geoms[geoms['keys']==closest_neighbor['keys']].iloc[0].name
                assert name_closest_geom != geom.name

                # merge the two geoms
                geom.geometry = geom.geometry.union(closest_neighbor.geometry)
                geom['coastal'] = True
                geom['merged'] = True

                # remove closest neighbor from geoms (will be replaced by merged geom)
                geoms = geoms.drop(name_closest_geom)
                # remove the now merged inland geom from geoms
                name_geom_to_remove = geoms[geoms['keys']==geom['keys']].iloc[0].name
                geoms = geoms.drop(name_geom_to_remove)

                # append merged geom to geoms
                geom_df = gpd.GeoDataFrame([geom], columns=geoms.columns, crs=geoms.crs)
                geoms = pd.concat([geoms, geom_df], ignore_index=True)

        iter+=1
        if iter > 20:
            print(f'Merge not completed, {len(geoms[geoms["coastal"]==False])} geoms left in country')
            country = geoms.iloc[0]['GID_0']
            geoms.to_file(f'DataDrive/government_geoms/government_geoms_merged_{country}.gpkg', driver='GPKG')
            break
    return geoms

                


def main():
    geoms = load_geoms('DataDrive/government_geoms/government_geoms.gpkg')
    coastline = load_coastline('DataDrive/coastal_defense/shoreline_landward.gpkg')

    # iterate over geoms and set attribute to true if it has a floodplain
    countries = np.unique(geoms['GID_0'])
    geoms_merged = gpd.GeoDataFrame(columns=geoms.columns, crs=geoms.crs)
    for country in countries:
        print(f'Processing {country}')
        geoms_country = geoms[geoms['GID_0'] == country]
        geoms_country_merged = iterate_over_geoms(geoms_country, coastline)
        geoms_merged = pd.concat([geoms_merged, geoms_country_merged], ignore_index=True)

    geoms_merged['idx'] = np.arange(len(geoms_merged))
    geoms_merged.to_file('DataDrive/government_geoms/government_geoms_merged.gpkg', driver='GPKG')
    rasterize_geoms(file='government_geoms_merged.gpkg')
    print('Done')

if __name__ == "__main__":
    # main()
    rasterize_geoms(file='government_geoms_merged.gpkg')