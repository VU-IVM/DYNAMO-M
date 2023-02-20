from DYNAMO_plots import plot_single_runs_supplement




if __name__ == '__main__':
    report_folder = 'report'
    admin_idx = ['FRA.7.2_1_flood_plain', 'FRA.10.1_1_flood_plain']
    out_file = 'figure_supplement.png'
    
    variables = {
        'population': {
            'scaling': 1,
            'title': '',
            'ylabel': 'Population within the floodzone',
            # 'ylims': [40_000, 18_000],
        },
        'ead_total': {
            'scaling': 1,
            'ylabel': 'Expected annual damages',
            'title': '',
        },
        'n_households_adapted': {
            'scaling': 1,
            'ylabel': 'Households adapted',
            'title': '',
    }
    }

    plot_single_runs_supplement(report_folder, admin_idx, variables, out_file, show = True)



