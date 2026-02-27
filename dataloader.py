from __future__ import annotations

from configreader import ConfigReader

import numpy as np
from pathlib import Path
from typing import List
from numpy.typing import NDArray

class DataLoader:
    def __init__(
        self,
        config: 'ConfigReader',
        directory: None | Path
) -> None:

        # Acces to configfile
        self.config = config

        # Principal path
        self.base_dir = Path(__file__).resolve().parent

        # Get data folders name from config
        unbiased_folder: str = self.config.get('UNBIASED_FOLDER', 'unbiased')
        biased_folder: str = self.config.get('BIASED_FOLDER', 'biased')

        # Get file extension
        self.file_extension: str = self.config.get('FILE_EXTENSION', '.dat')

        # Save results
        results_folder: str = self.config.get('RESULTS_FOLDER', 'results')

        if directory == None:

            # Path file dat to run analysis
            self.unbiased_path: Path = self.base_dir / unbiased_folder
            self.biased_path: Path = self.base_dir / biased_folder

            if not (self.base_dir / results_folder).exists():

                self.results_path: Path = Path(self.base_dir / results_folder)
                self.results_path.mkdir(parents=True, exist_ok=True)

            else:

                self.results_path: Path = self.base_dir / results_folder


        elif directory != None:

            self.subdirectory = directory

            # Path file dat to run analysis
            self.unbiased_path: Path = self.subdirectory / unbiased_folder
            self.biased_path: Path = self.subdirectory / biased_folder

            self.results_path: Path = self.subdirectory / results_folder

            # Check if the results directory exists on the disk
            if not self.results_path.exists():

                self.results_path.mkdir(parents=True, exist_ok=True)

            else:

                self.results_path = self.subdirectory / results_folder

        # Load voltage range
        voltage_start: int = self.config.get('VOLTAGE_START', -110)
        voltage_stop: int = self.config.get('VOLTAGE_STOP', 40)
        voltage_step: int = self.config.get('VOLTAGE_STEPS', 151)

        # Load voltages
        self.voltage: NDArray[np.float64] = np.linspace(voltage_start, voltage_stop, voltage_step)

    def discover_data_files(self, folderpath: Path) -> List[Path]:

        """Get full path of data by checking the existance"""

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
        """ Load data by removing corrupeted one """
        u_files = self.discover_data_files(self.unbiased_path)
        b_files = self.discover_data_files(self.biased_path)


        u_raw = [self._safe_load(f) for f in u_files]
        u_raw = [np.atleast_1d(d) for d in u_raw if d is not None]

        b_raw = [self._safe_load(f) for f in b_files]
        b_raw = [np.atleast_1d(d) for d in b_raw if d is not None]


        return u_raw, b_raw

    def _safe_load(self, filepath: Path) -> NDArray[np.float64] | None:

        """Check if file is valid and return data array"""

        # Check if file is empty
        if filepath.stat().st_size == 0:
            print('WARNING: filename {filepath.name} is empty.')
            return None

        try:
            current = self.load_data(filepath)

            # Checks avoiding corrupted currents
            if current.size == 0 or len(current) != len(self.voltage):
                print(f"WARNING: Missing data points {filepath.name} skipped.")
                return None

            if np.isnan(current).any():
                print(f"WARNING: Corrupted data {filepath.name} contains NaN values.")
                return None

            return current

        except Exception as e:

            print(f"Encountered error while loading {filepath.name}: {e}")
            return None

    @staticmethod
    def load_data(filename: str) -> NDArray[np.float64]:
        """Load single file .dat"""
        data = np.genfromtxt(filename, comments="#", skip_header=14, unpack=True)
        return data[1]
