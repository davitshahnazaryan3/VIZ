"""
Generates figures for static pushover analysis
"""
import matplotlib.pyplot as plt


class SPO:
    def __init__(self):

        # Default color patterns
        self.color_grid = ['#840d81', '#6c4ba6', '#407bc1', '#18b5d8', '#01e9f5',
                           '#cef19d', '#a6dba7', '#77bd98', '#398684', '#094869']

    def base_shear_vs_top_displacement(self, data):
        """
        Plots base shear vs top displacement
        :param data: tuple
        :return: figure object
        """
        fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
        plt.plot(data[0], data[1], color=self.color_grid[2])
        plt.xlim(0, max(data[0]))
        plt.ylim(0, max(data[1])+50)
        plt.xlabel('Top displacement [m]')
        plt.ylabel("Base Shear [kN]")
        plt.grid(True, which="major", axis='both', ls="--", lw=1.0)
        plt.grid(True, which="minor", axis='both', ls="--", lw=0.5)

        return fig
