import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from dataloader import DataLoader
from standardmethod import StandardMethod
from configreader import ConfigReader
from pathlib import Path

print(f'numpy version: {np.__version__}')
print(f'matplotlib version: {matplotlib.__version__}')
print('\n')

config = ConfigReader('config.txt')
minus_27V = config.get('MINUS_27V', -27)

def comparison(inputlist: list) -> None:

    if len(inputlist) > 1:

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

        plt.savefig('comparison_ni_curr.pdf', dpi=300)

        fig3, ax3 = plt.subplots(figsize=config.get('FIGSIZE', (12, 10)))
        fig4, ax4 = plt.subplots()

        mask = data['voltage'] <= 0


        for data in inputlist:

            ax3.plot(-data['voltage'][mask], data['dni/dv'], label=f"{data['label']}", lw = 2, ls = '-')
            ax4.scatter(data['label'], data['ni_minus27V'], label=f"{data['label']}", alpha=  0.7, marker = 'o', s = 4)


        ax3.set_xlabel(r'$E_x$ [eV]')
        ax3.set_ylabel('dni/dV')
        ax3.set_title('NI Dist Func')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.savefig('comparison_dist_func.pdf', dpi=300)

        ax4.set_ylabel('Current [nA]')
        ax4.set_title(f'NI Currents at {minus_27V}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.savefig('comparison_minus_27V.pdf', dpi=300)
        plt.show()


def analysis(directory: None | Path) -> dict:

    config = ConfigReader('config.txt')
    current_scale = config.get('CURRENT_SCALE', 1e9)
    minus_27V = config.get('MINUS_27V', -27)

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

    idx_minus27V = np.argmin(np.abs(voltage - minus_27V))
    nicurr_minus27V = nicurr[idx_minus27V]

    dni_dv = savgol_filter(nicurr_nc[mask_nc], window_length=11, polyorder=3, deriv=1, delta=1)
    dni_dv_smooth = savgol_filter(dni_dv, window_length=11, polyorder=3, delta=1)

    fig1, ax1 = plt.subplots()

    # Add plot properties
    ax1.set_xlabel('Voltage [V]')
    ax1.set_ylabel('Current [nA]')
    ax1.set_title(f'Negative Ion Current')
    ax1.grid(ls = '-.', alpha = 0.5)

    ax1.errorbar(voltage, nicurr, dnicurr, marker = '.', ls = '')

    plt.savefig(results_folder / 'ni_step.pdf')
    plt.close(fig1)

    fig2, ax2 = plt.subplots()

    ax2.set_xlabel(r'$E_{x}$ [eV]')
    ax2.set_ylabel('dni/dV')
    ax2.set_title(f'Negative Ion Dist Func')
    ax2.grid(ls = '-.', alpha = 0.5)

    ax2.plot(voltage[mask_nc], dni_dv_smooth, ls = '-', lw = 2)

    plt.savefig(results_folder / 'dist_func.pdf')
    plt.close(fig2)

    output: dict = {
            'label': loader.subdirectory.name,
            'voltage': loader.voltage,
            'nicurr': nicurr,
            'dnicurr': dnicurr,
            'ni_minus27V': nicurr_minus27V,
            'dni/dv': dni_dv_smooth
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

def check_directories(directories: list) -> list:

    ignored_folders: str = input()
    to_ignore: list = []

    for f in ignored_folders.split():

        if int(f) < len(directories):

            to_ignore.append(int(f))

        else:

            print('\n')
            print(f'WARNING: {f} not valid - ignored.')
            print('Insert the number again:')
            f_correct = input()
            to_ignore.append(int(f_correct))

    directories = [d for i, d in enumerate(directories) if i not in to_ignore]

    print('\n')
    print('Directories suxcefully ignored!')

    return directories

def main() -> int:

    base_path = Path('.')
    bad_names = ['__pycache__', '.git', 'results']
    directories = [x for x in base_path.iterdir() if x.is_dir() and x.name not in bad_names]

    esc: bool = False

    print('Loading directories...\n')

    if len(directories) == 0:

        print('WARNING: your folder is empty\n')
        print('No directory found.\n')

    directories.sort()

    for directory in directories:

        print(f'{directory.name} found.')

    unbiased_path = base_path / 'unbiased'
    biased_path = base_path / 'biased'

    if unbiased_path in directories and biased_path in directories:

        print('\n')
        print('Unbiased and Biased directories found in the main folder')
        print('Do you want to analyse them? [y/n]')

        control = input()

        if control == 'y':

            print('\n')
            print('Analysis of a single data set: ')
            print('No comparison will be done')
            process_folders(folders=None)

        elif control == 'n':

            directories.remove(unbiased_path)
            directories.remove(biased_path)

            print('\n')
            print('Biased and Unbiased folders ignored.')
            print('The following folders will be anlysed:')

            for directory in directories:

                print(f'{directory.name}')

            print('\n')
            print('Do you want to ignore any folder? [y/n]')

            control_folders: str = input()

            try:

                if control_folders == 'n':

                    print('\n')
                    print('No folders will be ignored')
                    print('Analysis...')

                    results_list = process_folders(folders=directories)
                    comparison(results_list)

                elif control_folders == 'y':

                    print('\n')
                    print('Insert the corresponding number of the folder to be ignored\n')
                    print('To ignore more directories separate numbers with a space')

                    for i, directory in enumerate(directories):

                        print(f'{i}) {directory.name}')

                    directories = check_directories(directories)

                    results_list = process_folders(folders=directories)
                    comparison(results_list)

            except Exception as e:
                print(f"Errorr during the analysis: {e}")
                import traceback
                traceback.print_exc()

    else:

        print('\n')

        for directory in directories:

            print(f'{directory.name} to be analysed.')

        print('\n')
        print('Do you want to ignore any folder? [y/n]')

        control_folders: str = input()

        if control_folders == 'n':

            print('\n')
            print('No folders will be ignored')
            print('Analysis...')

            results_list = process_folders(folders=directories)
            comparison(results_list)

        elif control_folders == 'y':

            print('\n')
            print('Insert the corresponding number of the folder to be ignored\n')
            print('To ignore more directories separate numbers with a space')

            for i, directory in enumerate(directories):

                print(f'{i}) {directory.name}')

            directories = check_directories(directories)

            results_list = process_folders(folders=directories)
            comparison(results_list)

    return 0

if __name__ == "__main__":
    main()
