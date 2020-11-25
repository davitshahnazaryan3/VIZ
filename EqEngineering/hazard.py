import matplotlib.pyplot as plt
import numpy as np


class Hazard:
    def __init__(self, period=None):
        """
        Initialize hazard
        :param period: float                Period of interest to highlight
        """
        self.period = period
        # Default color patterns
        self.color_grid = ['#840d81', '#6c4ba6', '#407bc1', '#18b5d8', '#01e9f5',
                           '#cef19d', '#a6dba7', '#77bd98', '#398684', '#094869']
        self.gray = "#c4c1c0"

    def true_hazard(self, data):
        """
        Initialize hazard
        :param data: pickle                 Hazard file (*.pickle)
        :return: figure object
        """
        periodIndex = int(self.period * 10)

        # Generate figure 1
        fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
        for i in range(len(data[0])):
            if i == 0:
                labelgray = "Hazard"
            else:
                labelgray = None
            plt.loglog(data[1][i], data[2][i], color=self.gray, ls="--", label=labelgray)

        plt.loglog(data[1][periodIndex], data[2][periodIndex], color="r", label=f"T={self.period}s")
        plt.ylim(10e-5 - 80e-6, 10e-1 + 0.1)
        plt.xlim(0.01, 1.1)
        plt.xlabel(r'Intensity Measure - $Sa(T*)$ [g]')
        plt.ylabel("Mean Annual Frequency of Exceedance")
        plt.grid(True, which="major", axis='both', ls="--", lw=1.0)
        plt.grid(True, which="minor", axis='both', ls="--", lw=0.5)
        plt.legend(frameon=False,
                   loc='upper right',
                   fontsize=10)

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
