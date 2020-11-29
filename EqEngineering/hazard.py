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

        if not self.rp:
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

    def fitted_hazard(self, data):
        """
        Initialize hazard
        :param data: pickle                 Hazard file (*.pickle)
        """
        # Placeholder
        pass
