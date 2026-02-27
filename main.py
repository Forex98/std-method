import numpy as np
import matplotlib.pyplot as plt
from dataloader import DataLoader
from standardmethod import StandardMethod
from configreader import ConfigReader
from pathlib import Path

def comparison(inputlist: list) -> None:

    if len(inputlist) > 1:

        config = ConfigReader('config.txt')

        fig2, ax2 = plt.subplots(figsize=config.get('FIGSIZE', (12, 10)))

        for data in inputlist:

            # Negative collector
            ax2.errorbar(data['voltage'], data['nicurr'], yerr=data['dnicurr'],
                         fmt='.', label=f"{data['label']}")

        ax2.set_xlabel('Voltage [V]')
        ax2.set_ylabel('Current [nA]')
        ax2.set_title('Negative Ion Current')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.savefig('comparison.pdf', dpi=300)
        plt.show()

def analysis(directory: None | Path) -> dict:

    config = ConfigReader('config.txt')
    current_scale = config.get('CURRENT_SCALE', 1e9)

    loader = DataLoader(config, directory)

    unbiased_currents_raw, biased_currents_raw = loader.load_all_data()
    results_folder = loader.results_path
    sm = StandardMethod(loader.voltage, unbiased_currents_raw, biased_currents_raw, results_folder, config)
    sm.create_plot()
    plt.show(block=True)

    ratio = sm.calculate_ratio()

    sm.sat_cur_analysis()
    sm.negative_collector(ratio)
    sm.positive_collector()
    voltage = sm.voltage
    nicurr_nc = sm.ni_avg_nc*current_scale
    dnicurr_nc = sm.ni_std_nc*current_scale
    nicurr_pc = sm.ni_avg_pc*current_scale
    dnicurr_pc = sm.ni_std_pc*current_scale

    mask_nc = voltage <= 0
    mask_pc = voltage > 0

    nicurr = np.concatenate((nicurr_nc[mask_nc], nicurr_pc[mask_pc]))
    dnicurr = np.concatenate((dnicurr_nc[mask_nc], dnicurr_pc[mask_pc]))

    fig, ax = plt.subplots()

    # Add plot properties
    ax.set_xlabel('Voltage [V]')
    ax.set_ylabel('Current [nA]')
    ax.set_title(f'Negative Ion Current')
    ax.grid(ls = '-.', alpha = 0.5)

    ax.errorbar(voltage, nicurr, dnicurr, marker = '.', ls = '')
    plt.savefig(results_folder / 'ni_step.pdf')
    plt.close(fig)

    output: dict = {
            'label': loader.subdirectory,
            'voltage': loader.voltage,
            'nicurr': nicurr,
            'dnicurr': dnicurr,
            }

    return output

def process_folders(folders: None | Path) -> list:

    results_list: list = []

    if folders == None:

        analysis(folders)
        print('Analysis concluded.')

    else:

        for folder in folders:

            data_analysed = analysis(folder)

            results_list.append(data_analysed)

    return results_list

def main() -> int:

    base_path = Path('.')
    bad_names = ['__pycache__', 'plots', 'results']
    directories = [x for x in base_path.iterdir() if x.is_dir() and x.name not in bad_names]


    if len(directories) == 0:

        print('WARNING: your folder is empty')
        print('No directory found.')

    for directory in directories:

        print(f'{directory.name} found.')

    unbiased_path = base_path / 'unbiased'
    biased_path = base_path / 'biased'

    if unbiased_path in directories and biased_path in directories:

        print('Unbiased and Biased directories')
        print('found in the main folder')
        print('Do you want to anlyse them? [y/n]')

        control = input()

        if control == 'y':

            process_folders(folders=None)

        elif control == 'n':

            directories.remove(unbiased_path)
            directories.remove(biased_path)

            print('Biased and Unbiased folders ignored.')
            print('The following folders will be anlysed:')

            for directory in directories:

                print(f'{directory.name}')

            results_list = process_folders(folders=directories)
            comparison(results_list)

    else:

        for directory in directories:

            print(f'{directory.name} to be analysed.')

        print('Proceed with the analysis? [y/n]')
        confirm = input()

        if confirm == 'y':

            results_list = process_folders(folders=directories)
            comparison(results_list)

    return 0

if __name__ == "__main__":
    main()
