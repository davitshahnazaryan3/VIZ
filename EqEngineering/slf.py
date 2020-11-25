import matplotlib.pyplot as plt
import os
import pickle
import re
import pandas as pd
import numpy as np


class SLF:
    def __init__(self, path, nst, n_to_plot=100, geometry=0, normalizeCost=1):
        """
        Plots graphs separately for each storey and Performance group (i.e. 3 groups x n storeys)
        :param path: str                    Directory of SLF output files
        :param nst: int
        :param n_to_plot: int               Number of scatter points to plot
        :param geometry: int
        :param normalizeCost: float
        """
        # Default color patterns
        self.color_grid = ['#840d81', '#6c4ba6', '#407bc1', '#18b5d8', '#01e9f5',
                           '#cef19d', '#a6dba7', '#77bd98', '#398684', '#094869']
        self.geometry = geometry
        self.path = path
        self.normalizeCost = normalizeCost
        self.nst = nst
        self.n_to_plot = n_to_plot
        # Keys for identifying structural and non-structural components
        self.S_KEY = 1
        self.NS_KEY = 2
        self.PERF_GROUP = ["PSD_S", "PSD_NS", "PFA_NS"]
        # Default direction of SLFs to use if "2D" Structure is being considered
        self.DIRECTION = 1

    def read_slfs(self):
        """
        Reads SLF files
        :return: figure objects
        """
        # Check whether a .csv or .pickle is provided
        for file in os.listdir(self.path):
            if not os.path.isdir(self.path / file) and (file.endswith(".csv") or file.endswith(".xlsx")):
                # Read the SLFs
                loss, edp_cols, edp_range = self.slfs_csv()
                # Generate figure/s
                figs = self.plot_csv(loss, edp_range, edp_cols)
                return figs, edp_cols
            
            elif not os.path.isdir(self.path / file) and (file.endswith(".pickle") or file.endswith(".pkl")):
                # Read the SLFs
                loss, edps = self.slfs_pickle()
                # Generate figure/s
                figs, names = self.plot_pickle(loss, edps)
                return figs, names

            else:
                raise ValueError("[EXCEPTION] Wrong SLF file format provided! Should be .csv or .pickle")

    def slfs_csv(self):
        """
        SLFs are read and ELRs per performance group are derived
        SLFs for both PFA- and PSD-sensitive components are lumped at storey level, and not at each floor
        :return: ndarray                            Losses (PSD_S, PSD_NS, PFA_NS), EDP names, EDP ranges
        """
        los, edp_cols, edps_array = None, None, None
        
        for file in os.listdir(self.path):
            if not os.path.isdir(self.path / file):
                # Read the file (needs to be single file)
                filename = self.path / file
                try:
                    df = pd.read_excel(io=filename)
                except:
                    df = pd.read_csv(filename)

                # Get the feature names
                columns = np.array(df.columns, dtype="str")

                # Subdivide SLFs into EDPs and Losses
                testCol = np.char.startswith(columns, "E")
                lossCol = columns[testCol]
                edpCol = columns[np.logical_not(testCol)]

                edps = df[edpCol]
                edps_array = edps.to_numpy(dtype=float)
                loss = df[lossCol].to_numpy(dtype=float)
                loss /= self.normalizeCost
                
                # EDP names
                edp_cols = np.array(edps.columns, dtype="str")

                return loss, edp_cols, edps_array

    def plot_csv(self, loss, edp_range, edps):
        """
        Plots SLFs provided via csv file format
        :param loss: ndarray
        :param edp_range: ndarray
        :param edps: ndarray
        :return: list of figure objects
        """
        figs = []
        for i in range(len(edps)):
            fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
            x = edp_range[:, i]
            y = loss[:, i]
            edp = edps[i][0:3]
            label = f"{edps[i][0:3]}, {edps[i][4]}, {edps[i][-1]} storey"
            if edp in ["IDR", "IDR NS", "IDR S", "PSD", "PSD NS", "PSD S"]:
                xlabel = r"Peak Storey Drift, (PSD), $\theta$ [%]"
                xlim = [0, max(x)]
            else:
                xlabel = r"Peak Floor Acceleration, (PFA), $a$ [g]"
                xlim = [0, 4.0]
            ylim = [0, 1.0]

            ax.plot(x, y, color=self.color_grid[2], label=label)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"Loss, $L$")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.grid(True, which="major", axis="both", ls="--", lw=1.0)
            ax.legend(frameon=False, loc='upper right', fontsize=10)
            figs.append(fig)
        return figs
    
    def slfs_pickle(self):
        """
        SLFs are read and ELRs per performance group are derived
        :return: dict                               SLFs and EDP ranges
        """
        '''
        Inputs:
            EAL_limit: In % or currency
            Normalize: True or False

        If EAL is in %, then normalize should be set to True, normalization based on sum(max(SLFs))
        If EAL is in currency, normalize is an option
        If normalize is True, then normalization of SLFs will be carried out

        It is important to note, that the choice of the parameters is entirely on the user, 
        as the software will run successfully regardless.
        '''
        '''
        Outputs files have 6 features:  slfs                Regressed storey loss functions
                                        costs               Sampled costs at each simulation
                                        regression          Regression function name (currently supports 2 version)
                                        fit_pars            Regression function parameters
                                        accuracy            Accuracy metrics (try not to overcrowd)
                                        edp                 EDP range (x axis), supports PSD and PFA
        '''
        # SLF output file naming conversion is important (disaggregation is based on that)
        # IPBSD currently supports use of 3 distinct performance groups
        # i.e. PSD_NS, PSD_S, PFA_NS
        # Initialize
        SLFs = {"Directional": {"PSD_NS": {}, "PSD_S": {}},
                "Non-directional": {"PFA_NS": {}, "PSD_NS": {}, "PSD_S": {}}}

        # Initialize EDP ranges
        pfa = None
        psd = None
        for file in os.listdir(self.path):
            if not os.path.isdir(self.path / file) and not file.endswith(".csv") and not file.endswith(".xlsx"):

                # Open slf file
                f = open(self.path / file, "rb")
                df = pickle.load(f)
                f.close()

                str_list = re.split("_+", file)
                # Test if "2d" structure is being considered only
                if self.geometry == 0:
                    if str_list[0][-1] == "1" or len(str_list) == 2:
                        # Perform the loop if dir1 or non-directional components
                        pass
                    else:
                        # Skip the loop
                        continue

                if len(str_list) == 2:
                    direction = None
                    non_dir = "Non-directional"
                else:
                    direction = str_list[0][-1]
                    non_dir = "Directional"
                edp = str_list[-1][0:3]

                if edp == "pfa":
                    story = str_list[0][-1]
                    for key in df.keys():
                        if not key.startswith("SLF"):
                            loss = df[key]["slfs"]["mean"]
                            SLFs[non_dir]["PFA_NS"][str(int(story) - 1)] = {"loss": loss,
                                                                            "edp": df[key]["fragilities"]["EDP"]}
                            try:
                                pfa = df[key]["fragilities"]["EDP"]
                            except:
                                pfa = df[key]["edp"]

                else:
                    story = str_list[-2][-1]
                    for key in df.keys():
                        if not key.startswith("SLF"):
                            if key == str(self.S_KEY):
                                # if key == str(s_key):
                                tag = "PSD_S"
                            elif key == str(self.NS_KEY):
                                tag = "PSD_NS"
                            else:
                                raise ValueError("[EXCEPTION] Wrong group name provided!")

                            if direction is not None:
                                if "dir" + direction not in SLFs[non_dir][tag].keys():
                                    SLFs[non_dir][tag]["dir" + direction] = {}
                                loss = df[key]["slfs"]["mean"]
                                SLFs[non_dir][tag]["dir" + direction].update({story: {"loss": loss,
                                                                                      "edp": df[key]["fragilities"][
                                                                                          "EDP"]}})

                            else:
                                loss = df[key]["slfs"]["mean"]
                                SLFs[non_dir][tag].update({story: {"loss": loss,
                                                                   "edp": df[key]["fragilities"]["EDP"]}})

                            try:
                                psd = df[key]["fragilities"]["EDP"]
                            except:
                                psd = df[key]["edp"]

        # SLFs should be exported for use in LOSS
        # SLFs are disaggregated based on story, direction and EDP-sensitivity
        # Next, the SLFs are lumped at each storey based on EDP-sensitivity
        # EDP range should be the same for each corresponding group
        # Create SLF functions based on number of stories
        slf_functions = {}
        edps = {"PFA": pfa, "PSD": psd}
        for group in self.PERF_GROUP:
            slf_functions[group] = {}
            # Add for zero floor for PFA sensitive group
            if group == "PFA_NS":
                slf_functions[group]["0"] = np.zeros(pfa.shape)
            for st in range(1, self.nst + 1):
                if group == "PFA_NS":
                    edp = pfa
                else:
                    edp = psd
                slf_functions[group][str(st)] = np.zeros(edp.shape)

        # Generating the SLFs for each Performance Group of interest at each storey level
        for i in SLFs:
            for j in SLFs[i]:
                if i == "Directional":
                    for k in SLFs[i][j]:
                        for st in SLFs[i][j][k]:
                            loss = SLFs[i][j][k][st]["loss"]
                            slf_functions[j][st] += loss / self.normalizeCost
                else:
                    for st in SLFs[i][j]:
                        loss = SLFs[i][j][st]["loss"]
                        slf_functions[j][st] += loss / self.normalizeCost

        return slf_functions, edps
    
    def plot_pickle(self, loss, edps):

        # Initialize list of figures
        figs = []
        names = []
        # For each performance group
        for group in loss:
            # Get the edp range
            if group[0:3] == "PFA":
                edp = edps["PFA"]
                storeyLabel = "floor"
            else:
                edp = edps["PSD"]
                storeyLabel = "storey"

            # At each storey/floor level
            for st in loss[group]:
                # Names
                names.append(f"{group}_{st}")
                # Factor to reduce the y axis values to more readible values
                factor = 10**3 if self.normalizeCost == 1 else 1.0
                # Losses
                y = loss[group][st] / factor
                # Start figure generation
                fig, ax = plt.subplots(figsize=(4, 3), dpi=200)

                label = f"{group[0:3]}, {group[4:]}, {st} {storeyLabel}"
                if group[0:3] in ["IDR", "IDR NS", "IDR S", "PSD", "PSD NS", "PSD S"]:
                    xlabel = r"Peak Storey Drift, (PSD), $\theta$"
                    xlim = [0, 0.05]
                else:
                    xlabel = r"Peak Floor Acceleration, (PFA), $a$ [g]"
                    xlim = [0, 4.0]
                ylim = [0, max(y) + 50.]

                ax.plot(edp, y, color=self.color_grid[2], label=label)
                ax.set_xlabel(xlabel)
                if factor != 1.0:
                    addLabel = r", $x10^3$"
                else:
                    addLabel = ""
                ax.set_ylabel(r"Loss, $L$" + addLabel)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.grid(True, which="major", axis="both", ls="--", lw=1.0)
                ax.legend(frameon=False, loc='upper right', fontsize=10)
                figs.append(fig)

        return figs, names


if __name__ == "__main__":
    from pathlib import Path
    path = Path.cwd().parents[1] / ".applications/case1/Output/slfoutput"
    slf = SLF(path, 5)
    loss = slf.read_slfs()

