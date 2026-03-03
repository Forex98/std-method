from __future__ import annotations

from configreader import ConfigReader
from dataloader import DataLoader

from pathlib import Path
from typing import Tuple, List, Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

class StandardMethod:
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
        self.fit_v_min = self.config.get('FIT_V_MIN', -100)
        self.fit_v_max = self.config.get('FIT_V_MAX', -80)

        # Define reference voltages
        self.minus_90V = self.config.get('MINUS_90V', -90)
        self.minus_27V = self.config.get('MINUS_27V', -27)
        self.plus_20V = self.config.get('PLUS_20V', 20)

        # Plotting settings
        self.figsize = self.config.get('FIGSIZE', (12, 10))
        self.current_scale = self.config.get('CURRENT_SCALE', 1e9)

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
        self.ni_avg_nc: NDArray[np.float64] | None = None
        self.ni_std_nc: NDArray[np.float64] | None = None
        self.ni_avg_pc: NDArray[np.float64] | None = None
        self.ni_std_pc: NDArray[np.float64] | None = None
        self.pi_avg: float = 0.0
        self.pi_std: float = 0.0

        # Save results
        self.results_path = results_path

    @property
    def unbiased_currents(self):
        return [
            cur for cur, ok in zip(self.unbiased_currents_raw, self.unbiased_active)
            if ok
        ]
    @property
    def biased_currents(self):
        return [
            cur for cur, ok in zip(self.biased_currents_raw, self.biased_active)
            if ok
        ]

    @staticmethod
    def load_data(filename: str) -> NDArray[np.float64]:
        """Load single file .dat"""
        data = np.genfromtxt(filename, comments="#", skip_header=14, unpack=True)
        return data[1]


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
            alpha = 0.7,
            label = f'U_{i+1}'
            )

            self.unbiased_lines.append(line)

        for i, bcurrent in enumerate(self.biased_currents):
            line, = ax2.plot(
            self.voltage,
            bcurrent*self.current_scale,
            marker = '.',
            ls = '',
            alpha = 0.7,
            label = f'B_{i+1}'
            )

            self.biased_lines.append(line)

        # Add plot properties
        ax1.set_xlabel('Voltage [V]')
        ax1.set_ylabel('Current [nA]')
        ax1.set_title('Unbiased Currents')
        ax1.grid(ls = '-.', alpha = 0.5)

        # Add plot properties
        ax2.set_xlabel('Voltage [V]')
        ax2.set_ylabel('Current [nA]')
        ax2.set_title('Biased Currents')
        ax2.grid(ls = '-.', alpha = 0.5)

        # Add title
        pathsplitted = self.results_path.parts
        title = pathsplitted[0]
        fig.suptitle(f'{title}', fontweight='bold')

        self._add_checkboxes(fig)
        fig.tight_layout()

        figurename = self.results_path / 'I-V-characteristic.pdf'
        plt.savefig(figurename, dpi=300)


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

    def scale_curve(
        self,
        cur_unbiased: NDArray[np.float64],
        ratio: float | None = None
    )  -> NDArray[np.float64]:

        """
        Shift unbiased curve to match the biased one
        in the specified range
        """
        if ratio is None:
            raise ValueError("Ratio must be provided.")

        return cur_unbiased*ratio

    def negative_collector(self, ratio: NDArray[np.float64]) -> None:
        """
        Subtract any shifted unbiased curve to any biased one to measure the negative ion current
        """

        mask = (self.voltage > self.fit_v_min) & ( self.voltage < self.fit_v_max)
        ratio_masked = ratio[mask]
        # ratio avg
        ratio_avg = np.mean(ratio_masked)

        all_ni_results: List[float64] = []
        pi_values: List[float64] = []
        idx_90v = np.argmin(np.abs(self.voltage - (self.minus_90V)))

        for bcurrent in self.biased_currents:

            pi_values.append(bcurrent[idx_90v])

            for ucurrent in self.unbiased_currents:

                # Shift and subtract unbiased currents
                c_unb_scaled = self.scale_curve(ucurrent, ratio_avg)
                all_ni_results.append(bcurrent - c_unb_scaled)

        # Saving results
        self.ni_avg_nc = np.mean(all_ni_results, axis=0)
        self.ni_std_nc = np.std(all_ni_results, axis=0, ddof = 1)
        self.pi_avg = np.mean(pi_values)
        self.pi_std = np.std(pi_values)

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
        self.ni_avg_pc = np.mean(all_ni_results_pc, axis=0)
        self.ni_std_pc = np.std(all_ni_results_pc, axis=0, ddof = 1)


    def calculate_ratio(self) -> NDArray[np.float64]:

        # Averaging currents [nA]
        self.ucurrent_avg = np.mean(self.unbiased_currents, axis = 0)*self.current_scale
        self.bcurrent_avg = np.mean(self.biased_currents, axis = 0)*self.current_scale

        # Calculate the ratio
        ratio: NDArray[np.float64] = self.bcurrent_avg/self.ucurrent_avg

        # Plot the ratio vs voltage
        fig, ax = plt.subplots()
        mask = (self.voltage<0)
        ax.scatter(self.voltage[mask], ratio[mask], marker = '.')
        ax.set_xlabel('Voltage [V]')
        ax.set_ylabel('Ratio')
        ax.set_ylim(0, 1)
        ax.set_title(r'$I_b/I_u$')
        ax.grid(ls = '-.', alpha = 0.5)

        figurename_ratio = self.results_path / 'ratio.pdf'
        plt.savefig(figurename_ratio, dpi=300)
        plt.close(fig)

        return ratio

    @staticmethod
    def straight_line(
        x: NDArray[np.float64],
        m: float,
        q: float
    ) -> NDArray[np.float64]:

        return m*x + q

    def sat_cur_analysis(self) -> None:

        # Define the region of interest
        mask = (self.voltage < self.fit_v_max)
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
        ratio_avg = np.mean(ratio[19:39])
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
        ax2.grid(ls = '-.', alpha = 0.4)
        #ax2.legend(loc='lower right', frameon=False)

        fig.tight_layout()

        figurename_saturation = self.results_path / 'sat_cur_analysis.pdf'
        plt.savefig(figurename_saturation, dpi=300)
        plt.close(fig)
    # def get_variabilities(self):
    #
    #     # Calculate variability of the slope at -27V
    #     values_slope_ratio = []
    #     values_slope_shift = []
    #     values_ni_ratio = []
    #     values_ni_shift = []
    #
    #     for bcurrent in self.biased_currents:
    #
    #         for ucurrent in self.unbiased_currents:
    #
    #             # Shift and subtract unbiased currents
    #             ratio = bcurrent/ucurrent
    #             ratio_avg = np.mean(ratio[self.fit_v_min:self.fit_v_max])
    #             uscaled = ratio_avg*ucurrent
    #             ushifted = self.shift_curve(
    #             ucurrent,
    #             bcurrent,
    #             self.v_min,
    #             v_max = self.v_max
    #             )
    #             nicurrent_ratio = bcurrent - uscaled
    #             nicurrent_shift = bcurrent - ushifted
    #
    #             values_slope_ratio.append(uscaled[self.minus_27V]*self.current_scale)
    #             values_slope_shift.append(ushifted[self.minus_27V]*self.current_scale)
    #             values_ni_ratio.append(nicurrent_ratio[self.minus_27V]*self.current_scale)
    #             values_ni_shift.append(nicurrent_shift[self.minus_27V]*self.current_scale)
    #
    #     array_slope_ratio = np.array(values_slope_ratio)
    #     array_slope_shift = np.array(values_slope_shift)
    #     array27 = np.ones_like(array_slope_ratio)*(self.minus_27V)
    #
    #     fig = plt.figure(layout='constrained', figsize=self.figsize)
    #     ax_array = fig.subplots(2, 2, squeeze=False)
    #
    #
    #     ax_array[0, 0].scatter(array27, array_slope_ratio)
    #     ax_array[0, 0].set_xlabel('Voltage [V]')
    #     ax_array[0, 0].set_ylabel('Current [nA]')
    #     ax_array[0, 0].set_title('Slope varibility of scaled unbiased currents at -27V')
    #     ax_array[0, 0].grid(ls = '-.', alpha = 0.4)
    #
    #     ax_array[0, 1].scatter(array27, array_slope_shift)
    #     ax_array[0, 1].set_xlabel('Voltage [V]')
    #     ax_array[0, 1].set_ylabel('Current [nA]')
    #     ax_array[0, 1].set_title('Slope varibility of shifted unbiased currents at -27V')
    #     ax_array[0, 1].grid(ls = '-.', alpha = 0.4)
    #
    #     ax_array[1, 0].scatter(array27, values_ni_ratio)
    #     ax_array[1, 0].set_xlabel('Voltage [V]')
    #     ax_array[1, 0].set_ylabel('Current [nA]')
    #     ax_array[1, 0].set_title('Variability - Subtraction - Scaled -27V')
    #     ax_array[1, 0].grid(ls = '-.', alpha = 0.4)
    #
    #     ax_array[1, 1].scatter(array27, values_ni_shift)
    #     ax_array[1, 1].set_xlabel('Voltage [V]')
    #     ax_array[1, 1].set_ylabel('Current [nA]')
    #     ax_array[1, 1].set_title('Variability - Subtraction - Shift -27V')
    #     ax_array[1, 1].grid(ls = '-.', alpha = 0.4)
