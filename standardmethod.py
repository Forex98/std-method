from __future__ import annotations

##
# @file standardmethod.py
# @brief Implements the StandardMethod class for loading, plotting, scaling,
#        and analyzing current-voltage characteristics.
#
# This module provides utilities to compare unbiased and biased current
# acquisitions, compute scaling ratios, estimate negative ion currents,
# and generate diagnostic plots for the analysis workflow.

from configreader import ConfigReader
from dataloader import DataLoader

from pathlib import Path
from typing import Tuple, List, Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

##
# @class StandardMethod
# @brief Encapsulates the standard analysis workflow for current-voltage data.
#
# The class stores raw unbiased and biased current acquisitions, allows the user
# to interactively disable bad measurements, computes averaged currents and
# scaling ratios, and derives negative ion current estimates for different
# collector configurations.
class StandardMethod:
    ##
    # @brief Construct a StandardMethod analysis object.
    #
    # @param voltage Voltage array associated with all current measurements.
    # @param unbiased_currents_raw List of raw unbiased current arrays.
    # @param biased_currents_raw List of raw biased current arrays.
    # @param results_path Directory where output plots and results are saved.
    # @param config Configuration reader used to retrieve analysis parameters.
    def __init__(
        self,
        voltage: NDArray[np.float64],
        unbiased_currents_raw: List[NDArray[np.float64]],
        biased_currents_raw: List[NDArray[np.float64]],
        results_path: Path,
        config: 'ConfigReader'
) -> None:

        # Fundamental attributes
        self.config = config
        self.voltage = voltage
        self.unbiased_currents_raw = unbiased_currents_raw
        self.biased_currents_raw = biased_currents_raw

        # Define shifting range for analysis
        self.v_min = self.config.get('V_MIN', -90)
        self.v_max = self.config.get('V_MAX', -70)

        # Define scaling range for analysis
        self.scaling_v_min = self.config.get('SCALING_V_MIN', -100)
        self.scaling_v_max = self.config.get('SCALING_V_MAX', -80)

        # Define reference voltages
        self.minus_90V = self.config.get('MINUS_90V', -90)
        self.minus_40V = self.config.get('MINUS_40V', -40)

        # Plotting settings
        self.figsize = self.config.get('FIGSIZE', (12, 10))
        self.current_scale: str = self.config.get('CURRENT_SCALE', 1e9)
        self.grid_transparency: float = self.config.get('GRID_TRANSPARENCY', 0.7)
        self.grid_linestyle: str = self.config.get('GRID_LINESTYLE', 'dashdot')

        # Active mask
        self.unbiased_active: List[bool] = [True]*len(self.unbiased_currents_raw)
        self.biased_active: List[bool] = [True]*len(self.biased_currents_raw)

        # Plot handles
        self.unbiased_lines = []
        self.biased_lines = []

        # Avg currents
        self.ucurrent_avg: NDArray[np.float64] | None = None
        self.bcurrent_avg: NDArray[np.float64] | None = None

        # Outputs
        self.ni_avg_nc_scaling: NDArray[np.float64] | None = None
        self.ni_std_nc_scaling: NDArray[np.float64] | None = None
        self.all_ni_minus_40V_scaling: NDArray[np.float64] | None = None
        self.all_slopes_minus_40V_scaling: NDArray[np.float64] | None = None
        self.ni_avg_nc_shifting: NDArray[np.float64] | None = None
        self.ni_std_nc_shifting: NDArray[np.float64] | None = None
        self.all_ni_minus_40V_shifting: NDArray[np.float64] | None = None
        self.all_slopes_minus_40V_shifting: NDArray[np.float64] | None = None
        self.ni_avg_pc: NDArray[np.float64] | None = None
        self.ni_std_pc: NDArray[np.float64] | None = None
        self.pi_avg: float = 0.0
        self.pi_std: float = 0.0

        # Save results
        self.results_path = results_path

    ##
    # @brief Return the list of active unbiased current acquisitions.
    #
    # Only currents whose corresponding activity flag is set to True are
    # returned.
    #
    # @return List of enabled unbiased current arrays.
    @property
    def unbiased_currents(self):
        return [
            cur for cur, ok in zip(self.unbiased_currents_raw, self.unbiased_active)
            if ok
        ]

    ##
    # @brief Return the list of active biased current acquisitions.
    #
    # Only currents whose corresponding activity flag is set to True are
    # returned.
    #
    # @return List of enabled biased current arrays.
    @property
    def biased_currents(self):
        return [
            cur for cur, ok in zip(self.biased_currents_raw, self.biased_active)
            if ok
        ]

    ##
    # @brief Load a single current trace from a .dat file.
    #
    # The file is read with NumPy, skipping the first 14 header lines and
    # ignoring lines beginning with '#'. The second unpacked column is returned.
    #
    # @param filename Path to the input .dat file.
    # @return Loaded current array.
    @staticmethod
    def load_data(filename: str) -> NDArray[np.float64]:
        """Load single file .dat"""
        data = np.genfromtxt(filename, comments="#", skip_header=14, unpack=True)
        return data[1]

    ##
    # @brief Create and save the current-voltage plots for unbiased and biased data.
    #
    # The method generates a two-panel figure, one for unbiased currents and one
    # for biased currents. Interactive checkboxes are added to allow hiding or
    # showing individual acquisitions.
    #
    # @return None
    def create_plot(self) -> None:
        """Plot any current"""

        # Create plot frame
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        #self.fig = fig

        self.unbiased_lines = []
        self.biased_lines = []

        # Plot unbiased currents
        for i, ucurrent in enumerate(self.unbiased_currents):
            line, = ax1.plot(
            self.voltage,
            ucurrent*self.current_scale,
            marker = '.',
            ls = '',
            alpha = self.grid_transparency,
            label = f'U_{i+1}'
            )

            self.unbiased_lines.append(line)

        for i, bcurrent in enumerate(self.biased_currents):
            line, = ax2.plot(
            self.voltage,
            bcurrent*self.current_scale,
            marker = '.',
            ls = '',
            alpha = self.grid_transparency,
            label = f'B_{i+1}'
            )

            self.biased_lines.append(line)

        # Add plot properties
        ax1.set_xlabel('Voltage [V]')
        ax1.set_ylabel('Current [nA]')
        ax1.set_title('Unbiased Currents')
        ax1.grid(ls = self.grid_linestyle, alpha = self.grid_transparency)

        # Add plot properties
        ax2.set_xlabel('Voltage [V]')
        ax2.set_ylabel('Current [nA]')
        ax2.set_title('Biased Currents')
        ax2.grid(ls = self.grid_linestyle, alpha = self.grid_transparency)

        # Add title
        pathsplitted = self.results_path.parts
        title = pathsplitted[0]
        fig.suptitle(f'{title}', fontweight='bold')

        self._add_checkboxes(fig)
        fig.tight_layout()

        figurename = self.results_path / 'I-V-characteristic.pdf'
        plt.savefig(figurename, dpi=300)

    ##
    # @brief Add interactive checkboxes to enable or disable plotted acquisitions.
    #
    # Two checkbox groups are created: one for unbiased current traces and one
    # for biased current traces. Toggling a checkbox updates the visibility of
    # the corresponding line in the figure.
    #
    # @param fig Matplotlib figure to which the checkbox widgets are added.
    # @return None
    def _add_checkboxes(self, fig: Any) -> None:
        """Create interactive legend to deactivate
            bad acquisitions"""

        # Unbiased checkboxes
        # Coordinate ax: [left, bottom, width, height]
        # Fractions of figure sizes
        ax_u = fig.add_axes([0.06, 0.6, 0.2, 0.2], frameon=False)
        labels_u = [l.get_label() for l in self.unbiased_lines]

        self.check_u = CheckButtons(ax_u, labels_u, self.unbiased_active)

        for i, (lab, line) in enumerate(zip(self.check_u.labels, self.unbiased_lines)):
            lab.set_color(line.get_color())
            lab.set_fontweight('bold')

        def toggle_unbiased(label):
            idx = labels_u.index(label)
            self.unbiased_active[idx] = not self.unbiased_active[idx]
            self.unbiased_lines[idx].set_visible(self.unbiased_active[idx])
            fig.canvas.draw_idle()

        self.check_u.on_clicked(toggle_unbiased)

        # Biased checkboxes
        ax_b = fig.add_axes([0.55, 0.6, 0.2, 0.2], frameon=False)
        labels_b = [l.get_label() for l in self.biased_lines]

        self.check_b = CheckButtons(ax_b, labels_b, self.biased_active)

        for i, (lab, line) in enumerate(zip(self.check_b.labels, self.biased_lines)):
            lab.set_color(line.get_color())
            lab.set_fontweight('bold')

        def toggle_biased(label):
            idx = labels_b.index(label)
            self.biased_active[idx] = not self.biased_active[idx]
            self.biased_lines[idx].set_visible(self.biased_active[idx])
            fig.canvas.draw_idle()

        self.check_b.on_clicked(toggle_biased)

    ##
    # @brief Scale an unbiased current curve by a given ratio.
    #
    # This method applies a multiplicative factor to an unbiased current trace.
    #
    # @param cur_unbiased Unbiased current array to be scaled.
    # @param ratio Scaling factor to apply.
    # @return Scaled unbiased current array.
    # @throws ValueError If ratio is not provided.
    def scale_curve(
        self,
        cur_unbiased: NDArray[np.float64],
        ratio: float | None = None
    )  -> NDArray[np.float64]:

        """
        Scale unbiased curve to match the biased one
        """
        if ratio is None:
            raise ValueError("Ratio must be provided.")

        return cur_unbiased*ratio

    ##
    # @brief Shift an unbiased current curve by a given voltage range.
    #
    # This method calculates the averaged distance between unbiased and biased traces.
    #
    # @param cur_unbiased Unbiased current array to be shifted.
    # @param cur_biased Biased current to be matched.
    # @param V_min Minimum value of the voltage window.
    # @param V_max Maximum value of the voltage window.
    # @return Shifted unbiased current array.
    def shift_curve(
        self,
        cur_unbiased: NDArray[np.float64],
        cur_biased: NDArray[np.float64],
        V_min: float = None,
        V_max: float = None
    )  -> NDArray[np.float64]:

        """
        Shift unbiased curve to match the biased one
        in the specified range
        """

        # Assign defaults if no value is provided
        if V_min is None:
            V_min = self.v_min
        if V_max is None:
            V_max = self.v_max

        mask = (self.voltage > V_min) & (self.voltage < V_max)
        cur_unbiased_masked: NDArray[np.float64] = cur_unbiased[mask]
        cur_biased_masked: NDArray[np.float64] = cur_biased[mask]

        shift: float = np.mean(cur_biased_masked - cur_unbiased_masked)

        return cur_unbiased + shift

    ##
    # @brief Compute negative ion current for the negative collector configuration.
    #
    # A mean scaling factor is computed over the fit region from the provided
    # ratio array. Each biased current is then compared with each scaled
    # unbiased current, and all resulting negative ion current estimates are
    # aggregated into mean and standard deviation arrays.
    #
    # The biased current at the configured -90 V reference point is also used to
    # estimate the positive ion current statistics.
    #
    # @param ratio Ratio array used to derive the average scaling factor.
    # @return None
    def negative_collector(self, ratio: NDArray[np.float64]) -> None:
        """
        Subtract any shifted unbiased curve to any biased one to measure the negative ion current
        """

        mask = (self.voltage > self.scaling_v_min) & ( self.voltage < self.scaling_v_max)
        ratio_masked = ratio[mask]

        ratio_avg = np.mean(ratio_masked)

        all_ni_results_scaling: List[float64] = []
        all_ni_results_shifting: List[float64] = []
        all_ni_minus_40V_shifting: list[np.float64] = []
        all_ni_minus_40V_scaling: list[np.float64] = []
        all_slopes_minus_40V_scaling: list[np.float64] = []
        all_slopes_minus_40V_shifting: list[np.float64] = []
        pi_values: List[float64] = []
        idx_90v = np.argmin(np.abs(self.voltage - (self.minus_90V)))
        idx_minus_40V: float = np.argmin(np.abs(self.voltage - (self.minus_40V)))

        for bcurrent in self.biased_currents:

            pi_values.append(bcurrent[idx_90v])

            for ucurrent in self.unbiased_currents:

                # Scale/shift and subtract unbiased currents
                c_unb_scaled = self.scale_curve(ucurrent, ratio_avg)
                all_slopes_minus_40V_scaling.append(c_unb_scaled[idx_minus_40V])
                all_ni_results_scaling.append(bcurrent - c_unb_scaled)
                all_ni_minus_40V_scaling.append(bcurrent[idx_minus_40V] - c_unb_scaled[idx_minus_40V])

                c_unb_shifted = self.shift_curve(ucurrent, bcurrent)
                all_slopes_minus_40V_shifting.append(c_unb_shifted[idx_minus_40V])
                all_ni_results_shifting.append(bcurrent - c_unb_shifted)
                all_ni_minus_40V_shifting.append(bcurrent[idx_minus_40V] - c_unb_shifted[idx_minus_40V])

        # Saving results
        self.ni_avg_nc_scaling = np.mean(all_ni_results_scaling, axis=0)
        self.ni_std_nc_scaling = np.abs(np.max(all_ni_results_scaling, axis=0) - np.min(all_ni_results_scaling, axis=0))/2
        self.all_slopes_minus_40V_scaling = np.array(all_slopes_minus_40V_scaling)
        self.all_ni_minus_40V_scaling = np.array(all_ni_minus_40V_scaling)

        self.ni_avg_nc_shifting = np.mean(all_ni_results_shifting, axis=0)
        self.ni_std_nc_shifting = np.abs(np.max(all_ni_results_shifting, axis=0) - np.min(all_ni_results_shifting, axis=0))/2
        self.all_slopes_minus_40V_shifting = np.array(all_slopes_minus_40V_shifting)
        self.all_ni_minus_40V_shifting = np.array(all_ni_minus_40V_shifting)

        self.pi_avg = np.mean(pi_values)
        self.pi_std = np.std(pi_values)

    ##
    # @brief Compare negative ion current obtained from scaling and shifting methods.
    #
    # The average negative ion (NI) current computed with the scaling approach is
    # compared to the one obtained with the shifting approach (both previously
    # produced by `negative_collector`). The method visualizes both curves as a
    # function of collector voltage and displays the uncertainty band associated
    # with the scaling method.
    #
    # The purpose is to assess the consistency between the two NI estimation
    # techniques by highlighting any discrepancy between their averaged results.
    # A figure showing both curves and the scaling uncertainty envelope is saved
    # to the results directory.
    #
    # The method also computes and returns the point-wise difference between the
    # NI currents derived from the scaling and shifting methods.
    #
    # @return diff Array containing the voltage-resolved difference between the
    #         NI currents from the scaling and shifting approaches.
    def compare_ni_methods(self) -> NDArray[np.float64]:
        """
        Compares NI current obtained via Scaling vs Shifting.
        Visualizes the discrepancy to check for equality.
        """
        if self.ni_avg_nc_scaling is None or self.ni_avg_nc_shifting is None:
            raise ValueError("Run negative_collector() first!")

        # Calculate absolute difference
        diff: NDArray[np.float64] = self.ni_avg_nc_scaling - self.ni_avg_nc_shifting

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.voltage, self.ni_avg_nc_scaling*self.current_scale, label='Scaling Method', color='green')
        ax.plot(self.voltage, self.ni_avg_nc_shifting*self.current_scale, label='Shifting Method', color='red', linestyle='--')
        ax.fill_between(self.voltage,
                        self.ni_avg_nc_scaling*self.current_scale - self.ni_std_nc_scaling*self.current_scale,
                        self.ni_avg_nc_scaling*self.current_scale + self.ni_std_nc_scaling*self.current_scale,
                        color='green', alpha=0.2)

        ax.set_title("NI Current Comparison: Scaling vs Shifting")
        ax.set_xlabel("Collector Voltage (V)")
        ax.set_ylabel("Current (nA)")
        ax.legend()
        ax.grid(alpha = self.grid_transparency, ls = self.grid_linestyle)

        figurename = self.results_path / 'ni_shift_vs_scale.pdf'

        plt.savefig(figurename, dpi=300)
        plt.close(fig)

        return diff

    def variabilities(self, bins: int = 8) -> None:
        """
        @brief Displays the statistical distribution of the NI measurements.

        This function plots and saves two histograms:
        - One for the NI distribution at -40V (scaling vs shifting).
        - One for the slope distribution at -40V (scaling vs shifting).

        If the data for scaling or shifting is missing, it will display a warning.
        """

        # @brief Check if data for scaling and shifting at -40V is available.
        if len(self.all_ni_minus_40V_scaling) == 0 or len(self.all_ni_minus_40V_shifting) == 0:
            print("\nUnsufficient data to compute the distributions.")
            print("\nPlease run negative_collector(ratio) firstly.")
            return

        # @brief Create a figure and axis for the NI distribution plot.
        fig, ax = plt.subplots(figsize=self.figsize)


        # @brief Plot histogram for the scaling method values.
        ax.hist(self.all_ni_minus_40V_scaling*self.current_scale, bins=bins, alpha=0.5,
                label='Scaling', color='blue', edgecolor='navy')

        # @brief Plot histogram for the shitig method values
        ax.hist(self.all_ni_minus_40V_shifting*self.current_scale, bins=bins, alpha=0.5,
                label='Shifting', color='orange', edgecolor='darkorange')

        # @brief Calculate the mean values for scaling and shifiting distribution.
        mean_sc = np.mean(self.all_ni_minus_40V_scaling)*self.current_scale
        mean_sh = np.mean(self.all_ni_minus_40V_shifting)*self.current_scale

        # @brief Add vertical lines indicating the mean for each distribution.
        ax.axvline(mean_sc, color='blue', linestyle='--', label=f'Mean Scaling: {mean_sc:.2e}')
        ax.axvline(mean_sh, color='orange', linestyle='--', label=f'Mean Shifting: {mean_sh:.2e}')

        # @brief Set the plot title and axis labels.
        ax.set_title(f"NI distribution at {self.minus_40V}V - {self.results_path.name}")
        ax.set_xlabel("NI Current [nA]")
        ax.set_ylabel("")
        ax.legend()
        ax.grid(axis='y', alpha=self.grid_transparency, ls = self.grid_linestyle)

        # @brief Adjust layout; save the plot to a file; close the figure.
        plt.tight_layout()
        plt.savefig(self.results_path / "comparison_distribution.png")
        plt.close(fig)

        # @brief Check if data for slope distribution is available.
        if len(self.all_slopes_minus_40V_scaling) == 0 or len(self.all_slopes_minus_40V_shifting) == 0:
            print("\nUnsufficient data to compute the distributions.")
            print("\nPlease run negative_collector(ratio) firstly.")
            return

        # @brief Create a second figure and axis for the slope distribution plot.
        fig2, ax2 = plt.subplots(figsize=self.figsize)

        # @brief Plot histogram for the scaling slopes.
        ax2.hist(self.all_slopes_minus_40V_scaling*self.current_scale, bins=bins, alpha=0.5,
                label='Scaling', color='blue', edgecolor='navy')

        # @brief Plot histogram for the shifting slopes.
        ax2.hist(self.all_slopes_minus_40V_shifting*self.current_scale, bins=bins, alpha=0.5,
                label='Shifting', color='orange', edgecolor='darkorange')

        # @brief Calculate the mean value for the scaling and shifting slopes distribution.
        mean_slope_sc = np.mean(self.all_slopes_minus_40V_scaling)*self.current_scale
        mean_slope_sh = np.mean(self.all_slopes_minus_40V_shifting)*self.current_scale

        # @brief Add vertical lines indicating the mean for each slope distribution.
        ax2.axvline(mean_slope_sc, color='blue', linestyle='--', label=f'Mean Scaling: {mean_slope_sc:.2e}')
        ax2.axvline(mean_slope_sh, color='orange', linestyle='--', label=f'Mean Shifting: {mean_slope_sh:.2e}')

        # @brief Set the plot title and axis labels for the slope distribution.
        ax2.set_title(f"Slope distribution at {self.minus_40V} - {self.results_path.name}")
        ax2.set_xlabel("NI Corrente[nA]")
        ax2.set_ylabel("")
        ax2.legend()
        ax2.grid(axis='y', alpha=self.grid_transparency, ls = self.grid_linestyle)

        # @brief Adjust layout; save the plot to a file; close the figure.
        plt.tight_layout()
        plt.savefig(self.results_path / "comparison_slope_distribution.png")
        plt.close(fig2)


    ##
    # @brief Compute negative ion current for the positive collector configuration.
    #
    # Each biased current is subtracted from each unscaled unbiased current and
    # the resulting current differences are combined into mean and standard
    # deviation arrays.
    #
    # @return None
    def positive_collector(self) -> None:

        """
        Subtract any unshifted unbiased curve to any biased one to
        measure the negative ion current
        """

        all_ni_results_pc: List[float64] = []

        for bcurrent in self.biased_currents:

            for ucurrent in self.unbiased_currents:

                all_ni_results_pc.append(bcurrent - ucurrent)

        # Saving results
        self.ni_avg_pc: NDArray[np.float64] = np.mean(all_ni_results_pc, axis=0)
        self.ni_std_pc: NDArray[np.float64] = np.std(all_ni_results_pc, axis=0, ddof = 1)

    ##
    # @brief Compute the average biased-to-unbiased current ratio.
    #
    # Average unbiased and biased currents are first computed, scaled according
    # to the configured current scale, and then divided element-wise to form the
    # ratio array. A diagnostic ratio plot is also saved to disk.
    #
    # @return Array of biased-to-unbiased current ratios.
    def calculate_ratio(self) -> NDArray[np.float64]:

        # Averaging currents [nA]
        self.ucurrent_avg = np.mean(self.unbiased_currents, axis = 0)*self.current_scale
        self.bcurrent_avg = np.mean(self.biased_currents, axis = 0)*self.current_scale

        # Calculate the ratio
        ratio: NDArray[np.float64] = self.bcurrent_avg/self.ucurrent_avg

        # Plot the ratio vs voltage
        fig, ax = plt.subplots()
        #mask = (self.voltage<0)
        ax.scatter(self.voltage, ratio, marker = '.')
        ax.set_xlabel('Voltage [V]')
        ax.set_ylabel('Ratio')
        #ax.set_ylim(0, 1)
        ax.set_title(r'$I_b/I_u$')
        ax.grid(ls = self.grid_linestyle, alpha = self.grid_transparency)

        figurename_ratio = self.results_path / 'ratio.pdf'
        plt.savefig(figurename_ratio, dpi=300)
        plt.close(fig)

        return ratio

    ##
    # @brief Linear function used for curve fitting.
    #
    # @param x Input x values.
    # @param m Slope of the line.
    # @param q Intercept of the line.
    # @return Evaluated linear function values.
    @staticmethod
    def straight_line(
        x: NDArray[np.float64],
        m: float,
        q: float
    ) -> NDArray[np.float64]:

        return m*x + q

    ##
    # @brief Perform saturation current analysis on averaged currents.
    #
    # This method fits the low-voltage tail of both averaged unbiased and biased
    # currents with a straight line, prints the fit parameters and uncertainties,
    # computes a scaled unbiased current using an average ratio, and generates a
    # two-panel diagnostic plot containing the fits and the derived ion current.
    #
    # @return None
    def sat_cur_analysis(self) -> None:

        # Define the region of interest
        mask = (self.voltage < self.scaling_v_max)
        x_fit = self.voltage[mask]
        ucurrent_fit = self.ucurrent_avg[mask]
        bcurrent_fit = self.bcurrent_avg[mask]

        # Fit unbiased left tail
        uparameters, ucovariance = curve_fit(self.straight_line, x_fit, ucurrent_fit)
        m0u, q0u = uparameters
        dmu, dqu = np.sqrt(np.diagonal(ucovariance))

        print('Best Fit Reuslts - unbiased tail')
        print('m0: %f +- %f' % (m0u, dmu))
        print('q0: %f +- %f' % (q0u, dqu))

        # Fit biased left tail
        bparameters, bcovariance = curve_fit(self.straight_line, x_fit, bcurrent_fit)
        m0b, q0b = bparameters
        dmb, dqb = np.sqrt(np.diagonal(bcovariance))

        print('Best Fit Reuslts - biased tail')
        print('m0: %f +- %f' % (m0b, dmb))
        print('q0: %f +- %f' % (q0b, dqb))

        ratio = self.bcurrent_avg/self.ucurrent_avg
        idx_scaling_v_min = np.argmin(np.abs(self.voltage - self.scaling_v_min))
        idx_scaling_v_max = np.argmin(np.abs(self.voltage - self.scaling_v_max))
        ratio_avg = np.mean(ratio[idx_scaling_v_min:idx_scaling_v_max])
        ucurrent_scaled = ratio_avg * self.ucurrent_avg

        # Construct the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Plotting best fit line with unbiased current
        ax1.plot(
        self.voltage,
        self.ucurrent_avg,
        color = 'blue',
        lw = 2,
        label = 'Unbiased AVG'
        )
        ax1.plot(
        self.voltage,
        self.straight_line(self.voltage, m0u, q0u),
        color = 'black',
        ls = '--',
        label = 'Fit Unbiased Tail'
        )

        # Plotting best fit line with biased current
        ax1.plot(
        self.voltage,
        self.bcurrent_avg,
        color = 'orange',
        lw = 2,
        label = 'Biased AVG'
        )
        ax1.plot(
        self.voltage,
        self.straight_line(self.voltage, m0b, q0b),
        color = 'black',
        ls = '-.',
        label = 'Fit Biased Tail'
        )

        # Plotting rescaled unbiased current
        ax1.plot(
        self.voltage,
        ucurrent_scaled,
        color = 'green',
        lw = 2,
        label = 'Unbiased Scaled'
        )

        # Add plot properties
        ax1.set_xlabel('Voltage [V]')
        ax1.set_ylabel('Current [nA]')
        ax1.grid(ls = '-.', alpha = 0.4)
        ax1.legend(loc='lower right', frameon=False)

        # Add best fit results
        text = (
            r'Unbiased fit:' '\n'
            f'm = {m0u:.3f} +- {dmu:.3f}' '\n'
            f'q = {q0u:.3f} +- {dqu:.3f}' '\n'
            r'Biased fit:' '\n'
            f'm = {m0b:.3f} +- {dmb:.3f}' '\n'
            f'q = {q0b:.3f} +- {dqb:.3f}' '\n'
        )
        ax1.text(
            0.02, 0.98, text,
            transform = ax1.transAxes,
            va='top',
            ha='left',
            fontsize=10,
            bbox=dict(boxstyle='round', fc='white', alpha=0.85)
        )

        ni = self.bcurrent_avg - ucurrent_scaled
        ax2.plot(self.voltage, ni)
        ax2.set_xlabel('Voltage [V]')
        ax2.set_ylabel('Current [nA]')
        ax2.grid(ls = self.grid_linestyle, alpha = self.grid_transparency)
        #ax2.legend(loc='lower right', frameon=False)

        fig.tight_layout()

        figurename_saturation = self.results_path / 'sat_cur_analysis.pdf'
        plt.savefig(figurename_saturation, dpi=300)
        plt.close(fig)
