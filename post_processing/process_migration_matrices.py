import pandas as pd
import os
import geopandas as gpd

def sum_matrices(report_folder, export = True):
    #initiate df
    summed_matrices = pd.DataFrame() 
    matrices = os.listdir(os.path.join(report_folder, 'migration_matrices'))
    matrices = [file for file in matrices if not file.endswith('summed')]
    # loop through all matrices and load and create pairs
    for matrix_fn in matrices:
        matrix = pd.read_csv(os.path.join(report_folder, 'migration_matrices', matrix_fn))
        
        # filter floodplains
        matrix = matrix[matrix['from'].apply(lambda x: x.endswith('flood_plain'))]
        matrix_pairs = pd.DataFrame({'pair': matrix['from'] + '__' + matrix['to'], 'flow': matrix['flow']})

        if summed_matrices.size == 0:
            summed_matrices = matrix_pairs
        else:
            summed_matrices = pd.concat([summed_matrices, matrix_pairs])
        
    summed_matrices = summed_matrices.groupby('pair').sum()
    summed_matrices['from'] = [region.split('__')[0] for region in list(summed_matrices.index)]
    summed_matrices['to'] = [region.split('__')[1] for region in list(summed_matrices.index)]
    summed_matrices = summed_matrices[['from', 'to', 'flow']].reset_index(drop=True)
    
    if export: summed_matrices.to_csv(os.path.join(report_folder, 'migration_matrices', 'migration_matrix_summed.csv'))
    return summed_matrices

def georeference_summed_matrix(report_folder, summed_matrix):
    # load geoms
    path = os.path.join('DataDrive', 'SLR', 'admin', 'can_flood_gadm_2_merged.shp')
    geodataframe = gpd.read_file(path)
    centroids_from = geodataframe.set_index('keys').loc[summed_matrix['from']].centroid
    centroids_to = geodataframe.set_index('keys').loc[summed_matrix['to']].centroid
    
    x_from = [coord.coords[0][0] for coord in centroids_from]
    y_from = [coord.coords[0][1] for coord in centroids_from]

    x_to = [coord.coords[0][0] for coord in centroids_to]
    y_to = [coord.coords[0][1] for coord in centroids_to]

    # export referenced matrix
    summed_matrix_georef = pd.DataFrame(
        {
        'from': summed_matrix['from'],
        'to': summed_matrix['to'],
        'x_from': x_from,
        'y_from': y_from,
        'x_to': x_to,
        'y_to': y_to,
        'flow': summed_matrix['flow']
        }
        )

    summed_matrix_georef.to_csv(os.path.join(report_folder, 'migration_matrices', 'migration_matrix_summed_geo.csv'))

if __name__ == '__main__':
    summed_matrix = sum_matrices('report')
    georeference_summed_matrix('report', summed_matrix)