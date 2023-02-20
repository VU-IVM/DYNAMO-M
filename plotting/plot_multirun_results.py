from DYNAMO_plots import plume_plot_paper

if __name__ == '__main__':
    multirun_path = 'DataDrive/MULTIRUNS'
    
    # rcps = {'control':
    #             {'title': 'Baseline-SSP2'},
    #         'rcp4p5':
    #             {'title': 'RCP4.5-SSP2'}, 
    #         'rcp8p5':
    #             {'title': 'RCP8.5-SSP5'}
    #         }
    
    rcps = {'control':
                {'title': 'Baseline'},
            'rcp4p5':
                {'title': 'RCP4.5'}, 
            'rcp8p5':
                {'title': 'RCP8.5'}
            }
    

    variables = {
        'population_total':
            {'scaling': 1E-3, 'ylabel': 'Population residing in the floodplain [thousands]', 'ylims': (240, 400)},

        # 'population_near_beach_total':
        #     {'scaling': 1E3, 'ylabel': 'Population residing near a beach [thousands]', 'ylims': (None, 100)},

        'ead_total':
            {'scaling': 1E-6, 'ylabel': 'Residential flood risk [mln/ year]', 'ylims': (100, 700)},
        }
    
    settings = {
        'NoMigration':
            {'color': 'grey', 'linestyle': '--'}, 
        'NoPerception':
            {'color': 'black', 'linestyle': '-'},
        'NoAdaptation':
             {'color': 'darkred', 'linestyle': '-'},
        'Full': 
            {'color': 'darkgreen', 'linestyle': '-'}}
    
    out_file = 'multirun.png'

    plume_plot_paper(multirun_path, rcps, variables, settings, out_file, set_fontsize=11)
