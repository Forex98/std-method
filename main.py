import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from dataloader import DataLoader
from standardmethod import StandardMethod
from configreader import ConfigReader
from pathlib import Path
from numpy import typing

print(f'numpy version: {np.__version__}')
print(f'matplotlib version: {matplotlib.__version__}')
print('\n')

config = ConfigReader('config.txt')
minus_40V = config.get('MINUS_40V', -40)
plus_20V = config.get('PLUS_20V', 20)
current_scale = config.get('CURRENT_SCALE', 1e9)

def comparison(inputlist: list) -> None:

    if len(inputlist) > 1:

        fig2, ax2 = plt.subplots(figsize=config.get('FIGSIZE', (12, 10)))
        fig3, ax3 = plt.subplots(figsize=config.get('FIGSIZE', (12, 10)))

        for data in inputlist:

            ax2.errorbar(data['voltage'], data['nicurr'], yerr=data['dnicurr'],
                         fmt='.', label=f"{data['label']}")

            mask = data['voltage'] <= 0

            ax3.plot(-data['voltage'][mask], data['dni/dv'], label=f"{data['label']}", lw = 2, ls = '-')

        ax2.set_xlabel('Voltage [V]')
        ax2.set_ylabel('Current [nA]')
        ax2.set_title('Negative Ion Current')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.savefig('comparison_ni_curr.pdf', dpi=300)

        print('\n')
        print('Comparison Negative Ions currents done!')

        ax3.set_xlabel(r'$E_x$ [eV]')
        ax3.set_ylabel('dni/dV')
        ax3.set_title('NI Dist Func')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.savefig('comparison_dist_func.pdf', dpi=300)

        print('\n')
        print('Comparison Negative Ions Energy Distribution done!')

        te_list: NDArray = np.array(config.get('ELECTRON_TEMPERATURES', []))
        ne_list: NDArray = np.array(config.get('ELECTRON_DENSITIES', []))
        ion_mass: float = config.get('ION_MASS')

        pressures: NDArray = np.array(config.get('PRESSURES'))
        powers: NDArray = np.array(config.get('POWERS'))

        ni_minus40V = [data['ni_minus40V'] for data in inputlist]
        dni_minus40V = [data['dni_minus40V'] for data in inputlist]
        ni_plus20V = [data['ni_plus20V'] for data in inputlist]
        dni_plus20V = [data['dni_plus20V'] for data in inputlist]

        if len(pressures) > 0:

            fig4, ax4 = plt.subplots(figsize=config.get('FIGSIZE', (12, 10)))

            ax4.set_ylabel('Current [nA]')
            ax4.set_title(f'NI Current vs pressure')
            ax4.grid(True, alpha=0.3)

            ax4.errorbar(pressures, ni_minus40V, yerr=dni_minus40V, label='neg coll', marker = 'o')
            ax4.errorbar(pressures, ni_plus20V, yerr=dni_plus20V, label="pos coll", marker = '*')

            ax4.legend()

            plt.savefig('comparison_minus_40V_plus_20V.pdf', dpi=300)

            print('\n')
            print('Comparison negative vs positive collector done.')

            if len(te_list) > 0 and len(ne_list) > 0:

                fig5, ax5 = plt.subplots(figsize=config.get('FIGSIZE', (12, 10)))

                ax5.set_xlabel(r'$\Phi_{PI}$ ')
                ax5.set_ylabel('Current [nA]')
                ax5.set_title(f'NI Current vs PI flux')
                ax5.grid(True, alpha=0.3)

                cross_section: float = config.get('CROSS_SECTION')
                distance: float = config.get('DISTANCE')
                boltzmann: float = config.get('BOLTZMANN')
                gas_temperature: float = config.get('GAS_TEMPERATURE')
                gas_density: NDArray[np.float64] = pressures / (boltzmann * gas_temperature)
                mean_free_path: NDArray = 1 / (gas_density * cross_section)
                correction: NDArray = np.exp(-distance / mean_free_path)
                pi_fluxes: NDArray = 0.6 * ne_list * np.sqrt(te_list/ion_mass)

                # Applying correction ni

                ni_minus40V_corrected = ni_minus40V / correction

                ax5.scatter(pi_fluxes, ni_minus40V_corrected)

                plt.savefig('comparison_NI_curr_vs_pi_flux_pressure.pdf', dpi=300)
                print(f'Positive Ion Fluc / m-2: {pi_fluxes}')
                print(f'Corrected Neg Ion Curr: {ni_minus40V_corrected}')

        elif len(powers) > 0:

            fig4, ax4 = plt.subplots(figsize=config.get('FIGSIZE', (12, 10)))

            ax4.set_ylabel('Current [nA]')
            ax4.set_title(f'NI Current vs power')
            ax4.grid(True, alpha=0.3)

            ax4.errorbar(powers, ni_minus40V, yerr=dni_minus40V, label='neg coll', marker = 'o')
            ax4.errorbar(powers, ni_plus20V, yerr=dni_plus20V, label="pos coll", marker = '*')

            ax4.legend()

            plt.savefig('comparison_minus_40V_plus_20V.pdf', dpi=300)

            if len(te_list) > 0 and len(ne_list) > 0:

                fig5, ax5 = plt.subplots(figsize=config.get('FIGSIZE', (12, 10)))

                ax5.set_xlabel(r'$\Phi_{PI}$ ')
                ax5.set_ylabel('Current [nA]')
                ax5.set_title(f'NI Current vs PI flux')
                ax5.grid(True, alpha=0.3)

                pi_fluxes = 0.6 * ne_list * np.sqrt(te_list/ion_mass)

                ax5.scatter(pi_fluxes, ni_minus40V)

        else:

            print('\n')
            print('WARNING: No pressures or powers in the config file')
            print('No comparison NI vs pressure/power will be done')
            print('\n')


        plt.show()


def analysis(directory: None | Path) -> None | dict:

    loader = DataLoader(config, directory)

    unbiased_currents_raw, biased_currents_raw = loader.load_all_data()
    results_folder = loader.results_path
    sm = StandardMethod(loader.voltage, unbiased_currents_raw, biased_currents_raw, results_folder, config)
    sm.create_plot()
    plt.show(block=True)

    ratio = sm.calculate_ratio()

    sm.sat_cur_analysis()
    sm.negative_collector(ratio)
    sm.compare_ni_methods()
    sm.positive_collector()
    voltage = sm.voltage
    nicurr_nc = sm.ni_avg_nc_scaling*current_scale
    dnicurr_nc = sm.ni_std_nc_scaling*current_scale
    nicurr_pc = sm.ni_avg_pc*current_scale
    dnicurr_pc = sm.ni_std_pc*current_scale

    mask_nc = voltage <= 0
    mask_pc = voltage > 0

    nicurr = np.concatenate((nicurr_nc[mask_nc], nicurr_pc[mask_pc]))
    dnicurr = np.concatenate((dnicurr_nc[mask_nc], dnicurr_pc[mask_pc]))

    idx_minus40V = np.argmin(np.abs(voltage - minus_40V))
    nicurr_minus40V = nicurr[idx_minus40V]
    dnicurr_minus40V = dnicurr[idx_minus40V]

    idx_plus20V = np.argmin(np.abs(voltage - plus_20V))
    nicurr_plus20V = nicurr[idx_plus20V]
    dnicurr_plus20V = dnicurr[idx_plus20V]

    dni_dv = savgol_filter(nicurr_nc[mask_nc], window_length=11, polyorder=2, deriv=1, delta=1)
    dni_dv_smooth = savgol_filter(dni_dv, window_length=11, polyorder=2, delta=1)

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

    if directory:

        output: dict = {
                'label': loader.subdirectory.name,
                'voltage': loader.voltage,
                'nicurr': nicurr,
                'dnicurr': dnicurr,
                'ni_minus40V': nicurr_minus40V,
                'dni_minus40V': dnicurr_minus40V,
                'ni_plus20V': nicurr_plus20V,
                'dni_plus20V': dnicurr_plus20V,
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
        print('Do you want to analyse them? [y/n]\n')

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
            print('Do you want to ignore any folder? [y/n]\n')

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
