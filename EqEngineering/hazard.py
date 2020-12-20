import matplotlib.pyplot as plt
import numpy as np


class Hazard:
    def __init__(self, period=None, rp=None):
        """
        Initialize hazard
        :param period: list(float)          Period of interest to highlight
        :param rp: list(float)              Return periods to highlight on the graphs
        """
        self.period = period
        self.rp = rp
        # Default color patterns
        self.color_grid = ['#840d81', '#6c4ba6', '#407bc1', '#18b5d8', '#01e9f5',
                           '#cef19d', '#a6dba7', '#77bd98', '#398684', '#094869']
        self.FONTSIZE = 10
        self.markers = ["o", "v", "^", "<", ">", "s", "*", "D", "+", "X", "p"]
        self.gray = "#c4c1c0"

    @staticmethod
    def add_text(ax, x, y, text, ha='center', va='center', rotation=None, size=10, color='k'):
        ax.text(x, y, text, ha=ha, va=va, rotation=rotation, fontsize=size, color=color)

    def true_hazard(self, data, plotall=False):
        """
        Initialize hazard
        :param data: pickle                 Hazard file (*.pickle)
        :param plotall: bool                Plot all hazard curves, or only the ones corresponding to T
        :return: figure object
        """

        # Generate figure
        fig, ax = plt.subplots(figsize=(5, 3), dpi=200)
        if plotall:
            for i in range(len(data[0])):
                if i == 0:
                    labelgray = "Hazard"
                else:
                    labelgray = None
                plt.loglog(data[1][i], data[2][i], color=self.gray, ls="--", label=labelgray)

        for i in range(len(self.period)):
            periodIndex = int(self.period[i] * 10)
            labelAdd = f"T={self.period[i]}s" if self.period[i] > 0.0 else "PGA"
            plt.loglog(data[1][periodIndex], data[2][periodIndex], color=self.color_grid[i],
                       label=f"PSHA, {labelAdd}", marker=self.markers[i])

        if self.rp:
            for i in self.rp:
                plt.plot([0.01, 10.1], [1/i, 1/i], ls='--', color=self.gray)
                self.add_text(ax, 0.011, 1.5/i, f'Return period: {i} years', color=self.gray, ha='left')
        plt.ylim(10e-6, 1)
        plt.xlim(0.01, 10.1)
        plt.xticks(np.array([0.01, 0.1, 1.0, 10]))
        plt.xlabel(r'Intensity Measure, $s$ [g]', fontsize=self.FONTSIZE)
        plt.ylabel(r"Annual probability of exceedance, $H$", fontsize=self.FONTSIZE)
        plt.rc('xtick', labelsize=self.FONTSIZE)
        plt.rc('ytick', labelsize=self.FONTSIZE)
        plt.grid(True, which="major", ls="--", lw=0.8, dashes=(5, 10))
        plt.legend(frameon=False, loc='upper right', fontsize=self.FONTSIZE, bbox_to_anchor=(1.5, 1))

        # Period range
        periodRange = [x.strip("PGA, SA()") for x in data[0]]
        periodRange[periodRange == ""] = 0.0
        periodRange = np.array(periodRange, dtype=float)

        return fig

    def fitted_hazard(self, data, coefs):
        """
        Initialize hazard
        :param data: pickle                 Hazard file (*.pickle)
        :param coefs: pickle                Coefficients of fitted hazard
        """
        # Placeholder
        pass

    def both_hazard(self, true, fitted, coefs):
        """
        Plots the hazard function
        :param true: pickle                 True hazard file (*.pickle)
        :param fitted: pickle               Fitted hazard file (*.pickle)
        :param coefs: pickle                Coefficients of fitted hazard
        :return: figure object
        """
        hazard_fit = fitted["hazard_fit"]
        hazard_s = fitted["s"]
        hazard_T = fitted["T"]
        im, s, apoe = true

        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        if self.period:
            for i in range(len(self.period)):
                # ID corresponding to the period
                idx = int(self.period[i] * 10)
                # Labels
                labelAdd = f"PSHA, T={self.period[i]:.2f}s" if self.period[i] > 0.0 else "PSHA, PGA"
                labelAdd_2nd = r"Sa(%.2fs), $2^{nd}$ order fit" % self.period[i] if self.period[i] > 0.0 \
                    else r"PGA, $2^{nd}$ order fit"
                labelAdd_1st = r"Sa(%.2fs), $1^{st}$ order fit" % self.period[i] if self.period[i] > 0.0 \
                    else r"PGA, $1^{st}$ order fit"

                # Tag for the hazard to select
                if self.period[i] > 0.0:
                    try:
                        hazardTag = f"SA({self.period[i]:.2f})"
                        apoeFit = np.array(hazard_fit[hazardTag])
                        coef = coefs[hazardTag]
                    except:
                        hazardTag = f"SA({self.period[i]:.1f})"
                        apoeFit = np.array(hazard_fit[hazardTag])
                        coef = coefs[hazardTag]
                else:
                    hazardTag = "PGA"
                    apoeFit = np.array(hazard_fit[hazardTag])
                    coef = coefs[hazardTag]

                # 1st order fits
                h1, hx, saT1x = self.linearFit(coef, hazard_s)

                # Plotting
                plt.scatter(s[idx], apoe[idx], color=self.color_grid[i], label=labelAdd, marker=self.markers[i])
                plt.loglog(hazard_s, apoeFit, color=self.color_grid[i], label=labelAdd_2nd)
                plt.loglog(hazard_s, h1, color=self.color_grid[i], label=labelAdd_1st, ls="--", lw=0.8)
                if i == 0:
                    labelScatter = "Fitting points"
                else:
                    labelScatter = None
                plt.scatter(saT1x, hx, marker='x', c='k', zorder=10, s=40, label=labelScatter)

        if self.rp:
            for i in self.rp:
                plt.plot([0.01, 10.1], [1/i, 1/i], ls='--', color=self.gray)
                self.add_text(ax, 0.011, 1.5/i, f'Return period: {i} years', color=self.gray, ha='left')

        plt.ylabel('Annual probability of\n exceedance, ' + r'$H$', fontsize=12)
        plt.xlabel(r'Intensity, $s$ [g]', fontsize=12)
        plt.ylim(10e-6, 1)
        plt.xlim(0.01, 10.1)
        plt.xticks([])
        plt.xticks(np.array([0.01, 0.1, 1.0, 10]))
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)
        plt.legend(frameon=False,
                   loc='upper right',
                   fontsize=12,
                   bbox_to_anchor=(1.55, 1))

        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True, which="major", ls="--", lw=0.4, dashes=(5, 10))

        return fig

    def linearFit(self, coef, sa):
        """
        1st order linear fit
        :param coef: pickle
        :param sa: ndarray
        :return: ndarrays
        """
        k0 = coef["k0"]
        k1 = coef["k1"]
        k2 = coef["k2"]
        TR = np.array([475, 10000])
        H = np.array([1 / tr for tr in TR])
        SaT1 = np.exp((-k1 + np.sqrt(k1 ** 2 - 4 * k2 * np.log(H / k0))) / 2 / k2)

        kl0 = abs((np.log(H[0]) - np.log(H[1])) / (np.log(SaT1[0]) - np.log(SaT1[1])))
        kl00 = H[0] / np.exp(-kl0 * np.log(SaT1[0]))

        H_linear = kl00 * np.exp(-kl0 * np.log(sa))

        return H_linear, H, SaT1
