from __future__ import annotations

from configreader import ConfigReader

import numpy as np
from pathlib import Path
from typing import List
from numpy.typing import NDArray

class DataLoader:
    """
    @class DataLoader
    @brief Class for handling the loading of experimental data.

    This class is responsible for:
    - reading paths and parameters from the configuration file;
    - locating data files in the specified folders;
    - loading valid data from the files;
    - discarding corrupted, empty, or inconsistent files;
    - generating the voltage vector associated with the data.
    """

    def __init__(
        self,
        config: 'ConfigReader',
        directory: None | Path
) -> None:
        """
        @brief Constructor of the DataLoader class.

        Initializes the paths of the biased and unbiased data folders,
        creates the results folder if it does not exist, and generates
        the voltage vector based on the configuration parameters.

        @param config Configuration object used to read parameters.
        @param directory Optional base directory from which data are read
                         and results are saved. If None, the directory of
                         the current file is used.
        """

        ## @brief Access to the configuration file.
        self.config = config

        ## @brief Main directory containing the current file.
        self.base_dir = Path(__file__).resolve().parent

        ## @brief Name of the folder containing unbiased data.
        unbiased_folder: str = self.config.get('UNBIASED_FOLDER', 'unbiased')

        ## @brief Name of the folder containing biased data.
        biased_folder: str = self.config.get('BIASED_FOLDER', 'biased')

        ## @brief Extension of the data files to load.
        self.file_extension: str = self.config.get('FILE_EXTENSION', '.dat')

        ## @brief Name of the folder where results will be stored.
        results_folder: str = self.config.get('RESULTS_FOLDER', 'results')

        if directory == None:

            ## @brief Path to the folder containing unbiased files.
            self.unbiased_path: Path = self.base_dir / unbiased_folder

            ## @brief Path to the folder containing biased files.
            self.biased_path: Path = self.base_dir / biased_folder

            if not (self.base_dir / results_folder).exists():

                ## @brief Path to the results folder; created if it does not exist.
                self.results_path: Path = Path(self.base_dir / results_folder)
                self.results_path.mkdir(parents=True, exist_ok=True)

            else:

                ## @brief Path to the existing results folder.
                self.results_path: Path = self.base_dir / results_folder


        elif directory != None:

            ## @brief User-specified subdirectory.
            self.subdirectory = directory

            ## @brief Path to the folder containing unbiased files.
            self.unbiased_path: Path = self.subdirectory / unbiased_folder

            ## @brief Path to the folder containing biased files.
            self.biased_path: Path = self.subdirectory / biased_folder

            ## @brief Path to the results folder.
            self.results_path: Path = self.subdirectory / results_folder

            ## @brief Check if the results folder exists on disk.
            if not self.results_path.exists():

                self.results_path.mkdir(parents=True, exist_ok=True)

            else:

                self.results_path = self.subdirectory / results_folder

        ## @brief Starting value of the voltage range.
        voltage_start: int = self.config.get('VOLTAGE_START', -110)

        ## @brief Ending value of the voltage range.
        voltage_stop: int = self.config.get('VOLTAGE_STOP', 40)

        ## @brief Number of steps in the voltage range.
        voltage_step: int = self.config.get('VOLTAGE_STEPS', 151)

        ## @brief Voltage vector generated linearly.
        self.voltage: NDArray[np.float64] = np.linspace(voltage_start, voltage_stop, voltage_step)

    def discover_data_files(self, folderpath: Path) -> List[Path]:
        """
        @brief Finds all valid data files inside a folder.

        Verifies that the folder exists and searches for all files with the
        configured extension. If no files are found, an exception is raised.

        @param folderpath Path of the folder to analyze.
        @return Sorted list of full paths to the discovered files.
        @exception FileNotFoundError Raised if the folder does not exist
                                     or contains no valid files.
        """

        # Check if loaded folder exists
        if not folderpath.exists():
            raise FileNotFoundError(
            f"Error: No folder named {folderpath.name} found."
            )

        # Load .dat files within folder
        files = sorted(list(folderpath.glob(f"*{self.file_extension}")))

        if not files:
            raise FileNotFoundError(
            f"Error: No files found in {folderpath.absolute}."
            )

        # distinguish biased - unbiased folder
        pathsplitted = folderpath.parts
        foldertype = pathsplitted[len(pathsplitted)-1]

        print(
            f'{foldertype} data loaded correctly: \n'
            )

        for f in files:

            print(
                f'{f.name}'
                )
        print('\n')

        return files

    def load_all_data(self) -> Tuple[List[NDArray], List[NDArray]]:
        """
        @brief Loads all unbiased and biased data, excluding corrupted files.

        Files are first discovered in their respective folders and then loaded
        safely using the _safe_load() method. Invalid data are discarded.

        @return Tuple containing:
                - list of valid unbiased arrays;
                - list of valid biased arrays.
        """
        u_files = self.discover_data_files(self.unbiased_path)
        b_files = self.discover_data_files(self.biased_path)


        u_raw = [self._safe_load(f) for f in u_files]
        u_raw = [np.atleast_1d(d) for d in u_raw if d is not None]

        b_raw = [self._safe_load(f) for f in b_files]
        b_raw = [np.atleast_1d(d) for d in b_raw if d is not None]


        return u_raw, b_raw

    def _safe_load(self, filepath: Path) -> NDArray[np.float64] | None:
        """
        @brief Loads a data file while verifying its validity.

        The method checks that the file:
        - is not empty;
        - has the expected length;
        - does not contain NaN values.

        If any problem is detected, the file is skipped and None is returned.

        @param filepath Path of the file to load.
        @return Data array if valid, otherwise None.
        """

        ## @brief Check if the file is empty.
        if filepath.stat().st_size == 0:
            print('WARNING: filename {filepath.name} is empty.')
            return None

        try:
            current = self.load_data(filepath)

            ## @brief Check that the data are not empty and have the expected length.
            if current.size == 0 or len(current) != len(self.voltage):
                print(f"WARNING: Missing data points {filepath.name} skipped.")
                return None

            ## @brief Check for NaN values in the loaded data.
            if np.isnan(current).any():
                print(f"WARNING: Corrupted data {filepath.name} contains NaN values.")
                return None

            return current

        except Exception as e:

            print(f"Encountered error while loading {filepath.name}: {e}")
            return None

    @staticmethod
    def load_data(filename: str) -> NDArray[np.float64]:
        """
        @brief Loads a single .dat file.

        The file is read while ignoring header lines and comments.
        The second column of the dataset is returned.

        @param filename Name or path of the file to load.
        @return NumPy array containing the values of the second column.
        """
        data = np.genfromtxt(filename, comments="#", skip_header=14, unpack=True)
        return data[1]
