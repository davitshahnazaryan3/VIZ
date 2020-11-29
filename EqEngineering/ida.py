"""
IDA plotter. Should be run before postprocessor
"""
import matplotlib.pyplot as plt


class IDA:
    def __init__(self):

        # Default color patterns
        self.color_grid = ['#840d81', '#6c4ba6', '#407bc1', '#18b5d8', '#01e9f5',
                           '#cef19d', '#a6dba7', '#77bd98', '#398684', '#094869']
        self.grayscale = ['#111111', '#222222', '#333333', '#444444', '#555555',
                          '#656565', '#767676', '#878787', '#989898', '#a9a9a9']
        self.FONTSIZE = 10
        self.markers = ["o", "v", "^", "<", ">", "s", "*", "D", "+", "X", "p"]

    def disp_vs_im(self, data):

        disp = data["mtdisp"]
        im = data["im_qtile"]

        fig, ax = plt.subplots(figsize=(5, 3), dpi=200)
        plt.plot(disp, im[2], color=self.grayscale[8], label="84th quantile", ls="-.")
        plt.plot(disp, im[1], color=self.grayscale[4], label="50th quantile", ls="--")
        plt.plot(disp, im[0], color=self.grayscale[0], label="16th quantile", ls="-")
        plt.xlim(0.0, max(disp))
        plt.ylim(0.0, max(im[2]) + 0.5)
        plt.xlabel("Top displacement, [m]", fontsize=self.FONTSIZE)
        plt.ylabel("Sa, [g]", fontsize=self.FONTSIZE)
        plt.rc('xtick', labelsize=self.FONTSIZE)
        plt.rc('ytick', labelsize=self.FONTSIZE)
        plt.grid(True, which="major", ls="--", lw=0.8, dashes=(5, 10))
        plt.legend(frameon=False, loc='upper right', fontsize=self.FONTSIZE, bbox_to_anchor=(1.5, 1))

        return fig
