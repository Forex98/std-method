"""
@file main.py
@brief Main analysis module.

This script performs the complete analysis workflow for the negative ion
current measurements. It loads data, processes it using the StandardMethod
analysis pipeline, generates plots, and optionally compares multiple datasets.

@section main_features Main features
- Automatic loading of experimental data
- Negative and positive collector current analysis
- Distribution function computation
- Comparison between multiple datasets
- Plot generation and PDF export

@section dependencies Dependencies
The module relies on the following components:
- DataLoader: handles loading experimental data
- StandardMethod: performs the main data analysis
- ConfigReader: reads configuration parameters from file

@section usage Typical usage
Run the script directly:

@code{.sh}
python main.py
@endcode

The program will automatically detect data folders and guide the user
through the analysis process.

@author
Alessandro Forese
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from dataloader import DataLoader
from standardmethod import StandardMethod
from configreader import ConfigReader
from pathlib import Path
from numpy import typing


config = ConfigReader('config.txt')
minus_40V: float = config.get('MINUS_40V', -40)
plus_20V: float = config.get('PLUS_20V', 20)
current_scale: float = config.get('CURRENT_SCALE', 1e9)
grid_transparency: float = config.get('GRID_TRANSPARENCY', 0.7)
grid_linestyle: str = config.get('GRID_LINESTYLE', 'dashdot')
method: str = config.get('METHOD')
c: float = config.get('SPEED_LIGHT', 3e8)

def comparison(inputlist: list) -> None:
    """
    @brief Compare results from multiple analysed datasets.

    This function generates comparison plots between different datasets,
    including negative ion current curves, distribution functions, and
    current values at a specific voltage.

    @param inputlist
        List of dictionaries returned by the `analysis()` function.
        Each dictionary contains processed data such as voltage,
        currents, uncertainties, and derived quantities.

    @note
    The function produces the following plots:
    - Negative ion current comparison
    - Distribution function comparison
    - Current comparison at the configured voltage (e.g. -27 V)

    Output figures are saved as PDF files.
    """

    if len(inputlist) > 1:

        fig2, ax2 = plt.subplots(figsize=config.get('FIGSIZE', (12, 10)))

        for data in inputlist:

            if data['voltage'] is not None and len(data['voltage']) > 0:
                ax2.scatter(data['voltage'], data['nicurr'],
                             marker='o', label=f'{data['label']}')
            else:
                print(f"WARNING: {data['label']} skipped - Data empty or missing values.")

        ax2.set_xlabel('Voltage [V]')
        ax2.set_ylabel('Current [nA]')
        ax2.set_title('NI currents comparison')
        ax2.grid(True, alpha=grid_transparency, ls = grid_linestyle)
        ax2.legend()

        try:
            fig2.savefig('comparison_ni_curr.pdf', dpi=300) # Ora funzionerebbe anche questo!
            print("NI currents correctly compared!")
        except Exception as e:
            print(f"Error while saving plot: {e}")


        fig3, ax3 = plt.subplots(figsize=config.get('FIGSIZE', (12, 10)))


        for data in inputlist:

            if data['voltage'] is not None and len(data['voltage']) > 0:
                mask = data['voltage'] < 0
                ax3.plot(-data['voltage'][mask], data['dni/dv'], label=f"{data['label']}", lw = 2, ls = '-')
            else:
                print(f"WARNING: {data['label']} skipped - Data empty or missing values.")

        ax3.set_xlabel(r'$E_x$ [eV]')
        ax3.set_ylabel('dni/dV')
        ax3.set_title('NI Dist Func')
        ax3.legend()
        ax3.grid(True, alpha=grid_transparency)

        try:
            fig3.savefig('comparison_ni_curr.pdf', dpi=300)
            print("Energy distributions correctly compared!")
        except Exception as e:
            print(f"Error while saving plot: {e}")


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
            ax4.set_xlabel('Pressure [Pa]')
            ax4.set_xticks(pressures)
            ax4.set_title(f'NI Current vs pressure')
            ax4.grid(True, alpha=grid_transparency, ls = grid_linestyle)

            ax4.errorbar(pressures, ni_minus40V, yerr=dni_minus40V, label='neg coll', marker = 'o')
            ax4.errorbar(pressures, ni_plus20V, yerr=dni_plus20V, label="pos coll", marker = '*')

            ax4.legend()

            plt.savefig('comparison_minus_40V_plus_20V.pdf', dpi=300)

            print('\n')
            print('Comparison negative vs positive collector done.')

            if len(te_list) > 0 and len(ne_list) > 0:

                fig5, ax5 = plt.subplots(figsize=config.get('FIGSIZE', (12, 10)))

                ax5.set_xlabel(r'$\Phi_{PI}$ $[m^{-2} s^{-1}]$')
                ax5.set_ylabel('Current [nA]')
                ax5.set_title(f'NI Current vs PI flux')
                ax5.grid(True, alpha=grid_transparency, ls=grid_linestyle)

                cross_section: float = config.get('CROSS_SECTION')
                distance: float = config.get('DISTANCE')
                boltzmann: float = config.get('BOLTZMANN')
                gas_temperature: float = config.get('GAS_TEMPERATURE')
                gas_density: NDArray[np.float64] = pressures / (boltzmann * gas_temperature)
                mean_free_path: NDArray = 1 / (gas_density * cross_section)
                correction: NDArray = np.exp(-distance / mean_free_path)
                pi_fluxes: NDArray = 0.6 * ne_list * np.sqrt(te_list*c**2/ion_mass)

                # Applying correction ni

                ni_minus40V_corrected = ni_minus40V / correction

                ax5.scatter(pi_fluxes, ni_minus40V_corrected)

                plt.savefig('comparison_NI_curr_vs_pi_flux_pressure.pdf', dpi=300)
                print(f'Positive Ion Fluc / m-2: {pi_fluxes}')
                print(f'Corrected Neg Ion Curr: {ni_minus40V_corrected}')

        elif len(powers) > 0:

            fig4, ax4 = plt.subplots(figsize=config.get('FIGSIZE', (12, 10)))

            ax4.set_ylabel('Current [nA]')
            ax4.set_xlabel('Power [W]')
            ax4.set_xticks(powers)
            ax4.set_title(f'NI Current vs power')
            ax4.grid(True, alpha=grid_transparency, ls=grid_linestyle)

            ax4.errorbar(powers, ni_minus40V, yerr=dni_minus40V, label='neg coll', marker = 'o')
            ax4.errorbar(powers, ni_plus20V, yerr=dni_plus20V, label="pos coll", marker = '*')

            ax4.legend()

            plt.savefig('comparison_minus_40V_plus_20V.pdf', dpi=300)

            if len(te_list) > 0 and len(ne_list) > 0:

                fig5, ax5 = plt.subplots(figsize=config.get('FIGSIZE', (12, 10)))

                ax5.set_xlabel(r'$\Phi_{PI}$ [m^{-2} s^{-1}]')
                ax5.set_ylabel('Current [nA]')
                ax5.set_title(f'NI Current vs PI flux')
                ax5.grid(True, alpha=grid_transparency, ls=grid_linestyle)

                pi_fluxes = 0.6 * ne_list * np.sqrt(te_list*c**2/ion_mass)

                ax5.scatter(pi_fluxes, ni_minus40V)

        else:

            print('\n')
            print('WARNING: No pressures or powers in the config file')
            print('No comparison NI vs pressure/power will be done')
            print('\n')


        plt.show()


def analysis(directory: None | Path) -> None | dict:
    """
    @brief Perform the analysis of a single dataset.

    This function loads the experimental data, applies the standard
    analysis pipeline, computes derived quantities, and produces plots.

    @param directory
        Directory containing the dataset to analyse.
        If `None`, the function assumes the standard folder structure.

    @return dict
        Dictionary containing the processed results:
        - `label`: dataset name
        - `voltage`: voltage values
        - `nicurr`: negative ion current
        - `dnicurr`: uncertainty on current
        - `ni_minus40V`: current at configured voltage
        - `dni/dv`: derivative of current (distribution function)

    @note
    The function also saves the following plots:
    - Negative ion current vs voltage
    - Distribution function

    Results are stored inside the automatically generated results folder.
    """

    # Retrieve the threshold voltages from config
    threshold_voltage_negative: int = config.get('THRESHOLD_VOLTAGE_NEGATIVE', -5)
    threshold_voltage_positive: int = config.get('THRESHOLD_VOLTAGE_POSITIVE', 10)

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
    sm.variabilities()
    sm.positive_collector()
    voltage = sm.voltage

    # Choose the method to apply
    if method == 'scaling':

        print(f"-> Applying SCALING method for analysis")

        nicurr_nc = sm.ni_avg_nc_scaling*current_scale
        dnicurr_nc = sm.ni_std_nc_scaling*current_scale

    elif method == 'shifting':

        print(f"-> Applying SHIFTING for analysis")

        nicurr_nc = sm.ni_avg_nc_shifting*current_scale
        dnicurr_nc = sm.ni_std_nc_shifting*current_scale

    nicurr_pc = sm.ni_avg_pc*current_scale
    dnicurr_pc = sm.ni_std_pc*current_scale

    mask_nc = voltage <= threshold_voltage_negative
    mask_pc = voltage >= threshold_voltage_positive

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
    ax1.grid(True, alpha=grid_transparency, ls=grid_linestyle)

    mask = (voltage <= threshold_voltage_negative) | (voltage >= threshold_voltage_positive)
    ax1.errorbar(voltage[mask], nicurr, dnicurr, marker = '.', ls = '')


    plt.savefig(results_folder / 'ni_step.pdf')
    plt.close(fig1)

    fig2, ax2 = plt.subplots()

    ax2.set_xlabel(r'$E_{x}$ [eV]')
    ax2.set_ylabel('dni/dV')
    ax2.set_title(f'Negative Ion Dist Func')
    ax2.grid(True, alpha=grid_transparency, ls=grid_linestyle)

    ax2.plot(voltage[mask_nc], dni_dv_smooth, ls = '-', lw = 2)

    plt.savefig(results_folder / 'dist_func.pdf')
    plt.close(fig2)

    if directory:

        output: dict = {
                'label': loader.subdirectory.name,
                'voltage': loader.voltage[mask],
                'nicurr': nicurr,
                'dnicurr': dnicurr,
                'ni_minus40V': nicurr_minus40V,
                'dni_minus40V': dnicurr_minus40V,
                'ni_plus20V': nicurr_plus20V,
                'dni_plus20V': dnicurr_plus20V,
                'dni/dv': dni_dv_smooth
                }

        return output

def process_folders(folders: None | list[Path]) -> list[dict]:
    """
    @brief Process one or multiple dataset folders.

    @param folders
        Path to the dataset directory or a list of directories.
        If `None`, a single dataset analysis is executed.

    @return list
        List of dictionaries containing analysed data returned
        by the `analysis()` function.

    @note
    This function is used internally to handle batch processing
    of multiple experimental datasets.
    """

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
    """
    @brief Remove selected directories from the analysis list.

    The user can specify which folders should be ignored during the
    analysis phase by entering their corresponding indices.

    @param directories
        List of dataset directories detected in the working folder.

    @return list
        Updated list of directories after removing the ignored ones.

    @note
    The function performs basic validation of the provided indices
    and warns the user if invalid numbers are entered.
    """

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
    """
    @brief Entry point of the analysis program.

    This function scans the working directory, detects available
    datasets, and manages the full analysis workflow through
    user interaction.

    The user can:
    - analyse a single dataset (biased/unbiased structure)
    - analyse multiple datasets
    - ignore selected folders
    - generate comparison plots

    @return int
        Exit status code (`0` if execution completes successfully).

    @note
    The function automatically detects dataset folders in the current
    working directory and excludes system folders such as:
    - `__pycache__`
    - `.git`
    - `results`

    Comparison plots are generated when multiple datasets are analysed.
    """

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
    print(f'numpy version: {np.__version__}')
    print(f'matplotlib version: {matplotlib.__version__}')
    print('\n')

    config = ConfigReader('config.txt')
    minus_27V = config.get('MINUS_27V', -27)
    main()
