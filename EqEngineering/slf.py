import matplotlib.pyplot as plt


class Slf:
    def __init__(self):
        pass

    def visualize_slf(self, filename, edps=None, showplot=False, sflag=False, n_to_plot=100):
        """
        Visualizing graphs for the SLF generator
        :param filename: str                    File name, e.g. '*\filename.extension'
        :param edps: list(str)                  EDPs as keys used for accessing data and plotting, e.g. IDR S, IDR, PFA,
                                                IDR NS, PFA NS
        :param showplot: bool                   Whether to plot the figures in the interpreter or not
        :param sflag: bool                      Whether to save the figures or not
        :param n_to_plot: int                   Number of simulations to plot
        """
        if edps is None:
            edps = ["IDR S", "IDR NS", "PFA NS"]

        filepath = self.directory / "client" / filename

        if filename.endswith(".pkl") or filename.endswith(".pickle"):
            plot_tag = filename.replace(".pkl", "")
            with open(filepath, 'rb') as file:
                data = pickle.load(file)
        else:
            raise ValueError("[EXCEPTION] Currently only pickle or pkl file format is accepted")

        for edp in edps:
            if "IDR" in edp or "IDR NS" in edp or "IDR S" in edp:
                try:
                    idr_range = data["IDR S"]["fragilities"]["EDP"]
                except:
                    idr_range = data["IDR"]["fragilities"]["EDP"]

        for edp in edps:
            if "PFA" in edp:
                try:
                    pfa_range = data["PFA NS"]["fragilities"]["EDP"]
                except:
                    pfa_range = data["PFA"]["fragilities"]["EDP"]

        #        # IDR sensitive structural and non-structural performance group SLFs
        #        if "IDR" in edps or "IDR NS" in edps or "IDR S" in edps:
        #            fig1, ax = plt.subplots(figsize=(4, 3), dpi=100)
        #            cnt = 0
        #            for key in data["SLFs"]:
        #                y = data["SLFs"][key]
        #                if "IDR" in key:
        #                    plt.plot(idr_range, y, color=self.color_grid[cnt], label=key)
        #                cnt += 1
        #            plt.xlabel('IDR')
        #            plt.ylabel('E(L | IDR)')
        #            plt.xlim(0, 0.2)
        #            plt.ylim(0, 1)
        #            plt.grid(True, which="major", axis="both", ls="--", lw=1.0)
        #            plt.legend(frameon=False, loc="upper right", fontsize=10)
        #            if not showplot:
        #                plt.close()
        #        else:
        #            fig1 = None
        #
        #        # PFA sensitive non-structural performance group SLFs
        #        if "PFA" in edps or "PFA NS" in edps:
        #            fig2, ax = plt.subplots(figsize=(4, 3), dpi=100)
        #            cnt = 0
        #            for key in data["SLFs"]:
        #                y = data["SLFs"][key]
        #                if "PFA" in key:
        #                    plt.plot(pfa_range, y, color=self.color_grid[cnt], label=key)
        #                cnt += 1
        #            plt.xlabel('PFA [g]')
        #            plt.ylabel('E(L | IDR)')
        #            plt.xlim(0, 10)
        #            plt.ylim(0, 1)
        #            plt.grid(True, which="major", axis="both", ls="--", lw=1.0)
        #            plt.legend(frameon=False, loc="upper right", fontsize=10)
        #            if not showplot:
        #                plt.close()
        #        else:
        #            fig2 = None
        #
        #        # Sample fragility function
        #        component = data[edps[0]]["fragilities"]["ITEMs"][4]
        #        fig3, ax = plt.subplots(figsize=(4, 3), dpi=100)
        #        cnt = 0
        #        for key in component.keys():
        #            plt.plot(idr_range, component[key], color=self.color_grid[cnt], label=key)
        #            cnt += 1
        #        plt.xlabel('IDR')
        #        plt.ylabel('Probability of exceeding DS')
        #        plt.xlim(0, 0.2)
        #        plt.ylim(0, 1)
        #        plt.grid(True, which="major", axis="both", ls="--", lw=1.0)
        #        plt.legend(frameon=False, loc="upper right", fontsize=10)
        #        if not showplot:
        #            plt.close()

        #        # EDP vs loss ratio
        #        for edp in edps:
        #            if edp in ["IDR", "IDR NS", "IDR S"]:
        #                edp_range = idr_range
        #                xlim = [0, 0.2]
        #                ylim = [0, 1.4]
        #                xlabel = edp[0:3]
        #            else:
        #                edp_range = pfa_range
        #                xlim = [0, 10.0]
        #                ylim = [0, 1.4]
        #                xlabel = edp[0:3] + " [g]"
        #
        #            fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        #            cnt = 0
        #            for key in data[edp]["edp_dv_fitted"].keys():
        #                y = data[edp]["edp_dv_fitted"][key]
        #                plt.plot(edp_range, y, color=self.color_grid[cnt], label=key)
        #                cnt += 2
        #            plt.xlabel(xlabel)
        #            plt.ylabel("Loss Ratio")
        #            plt.xlim(xlim)
        #            plt.ylim(ylim)
        #            plt.grid(True, which="major", axis="both", ls="--", lw=1.0)
        #            plt.legend(frameon=False, loc="best", fontsize=10)
        #            if not showplot:
        #                plt.close()
        #            if sflag:
        #                self.plot_as_emf(fig,filename=self.directory/"client"/'figures'/f'edp_loss_{edp}_'
        #                                                                                f'{plot_tag}'.replace(" ", "_"))
        #                self.plot_as_png(fig,filename=self.directory/"client"/'figures'/f'edp_loss_{edp}_'
        #                                                                                f'{plot_tag}'.replace(" ", "_"))

        # %% Loss in euro vs EDP (including the simulation scatter, the fractiles of
        #  the simulations and the fitted fractiles, and the fitted mean)
        # IDR-S
        #        idr_range = idr_range*100
        #        for edp in edps:
        #            component = data[edp]
        #
        #            fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        #            if edp == "IDR" or edp == "IDR NS" or edp == "IDR S":
        #                edp_range = idr_range
        #            else:
        #                edp_range = pfa_range
        #
        #            cnt = 0
        #            for key in component["edp_dv_euro"].keys():
        #                # Fractile loss curves, Fitted loss in Currency
        #                y_fit = component["edp_dv_euro"][key] / 10.0**3
        #                # Fractile loss curves not fitted
        #                y = component["losses"]["loss_curve"].loc[key] / 10.0**3
        #                # Plotting the unfitted fractiles
        #                plt.plot(edp_range, y, color=self.color_grid[cnt], label=key, alpha=0.5, marker='o', markersize=3)
        #                # Plotting the fitted fractiles
        #                plt.plot(edp_range, y_fit, color=self.color_grid[cnt], label=key)
        #                cnt +=2
        #
        #            # Sampled story losses at each simulation
        #            total_loss_storey = component["total_loss_storey"]
        #            # Generate a selection of random indices for plotting
        #            selection = np.random.randint(len(total_loss_storey), size=n_to_plot)
        #            loss_to_display = {}
        #            for sel in selection:
        #                loss_to_display[sel] = total_loss_storey[sel]
        #
        #            # Plotting the scatter points of the simulations/sampled story losses
        #            for key in loss_to_display.keys():
        #                y_scatter = loss_to_display[key] / 10.0**3
        #                plt.scatter(edp_range, y_scatter, edgecolors=self.color_grid[2], marker='o', s=3,
        #                            facecolors='none', alpha=0.5)
        #            # Assigning a label
        #            plt.scatter(edp_range, y_scatter, edgecolors=self.color_grid[2], marker='o', s=3,
        #                        facecolors='none', alpha=0.5, label="Simulations")
        #
        #            # Assignign labels and limits for axes
        #            if edp in ["IDR", "IDR NS", "IDR S"]:
        #                xlabel = edp[0:3] + " [%]"
        #                xlim = [0, 6.]
        #                ylim = [0, 400.]
        #            else:
        #                xlabel = edp[0:3] + " [g]"
        #                xlim = [0, 4.]
        #                ylim = [0, 400.]
        #
        #            plt.xlabel(xlabel)
        #            plt.ylabel(r"Losses [$10^3 €/100 m^2$]")
        #            plt.xlim(xlim)
        #            plt.ylim(ylim)
        #            plt.grid(True, which="major", axis="both", ls="--", lw=1.0)
        #            ax.legend(frameon=False, loc='center left', fontsize=10, bbox_to_anchor=(1, 0.5))
        #            # Showing the plot
        #            if not showplot:
        #                plt.close()
        #            # Storing the plots as .emf and .png
        #            if sflag:
        #                self.plot_as_emf(fig,filename=self.directory/"client"/'figures'/f'loss_{edp}_'
        #                                                                                f'{plot_tag}'.replace(" ", "_"))
        #                self.plot_as_png(fig,filename=self.directory/"client"/'figures'/f'loss_{edp}_'
        #                                                                                f'{plot_tag}'.replace(" ", "_"))
        #
        # %% Shared plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), sharey=True, dpi=100,
                                            gridspec_kw={'hspace': 0.1, 'wspace': 0.1})
        axes = [ax1, ax2, ax3]
        id_axis = 0
        if edp == "IDR" or edp == "IDR NS" or edp == "IDR S":
            idr_range = idr_range * 100

        for edp in edps:
            component = data[edp]
            if edp == "IDR" or edp == "IDR NS" or edp == "IDR S":
                edp_range = idr_range
            else:
                edp_range = pfa_range

            ax = axes[id_axis]
            cnt = 0
            for key in component["edp_dv_euro"].keys():
                if key == "mean":
                    lbl = "Mean"
                else:
                    lbl = f"{int(key * 100)}%"

                # Fractile loss curves, Fitted loss in Currency
                y_fit = component["edp_dv_euro"][key] / 10.0 ** 3
                # Fractile loss curves not fitted
                y = component["losses"]["loss_curve"].loc[key] / 10.0 ** 3
                # Plotting the unfitted fractiles
                ax.plot(edp_range, y, color=self.color_grid[cnt], label=lbl, alpha=0.5, marker='o', markersize=5)
                # Plotting the fitted fractiles
                ax.plot(edp_range, y_fit, color=self.color_grid[cnt], label=lbl)
                cnt += 2

            # Sampled story losses at each simulation
            total_loss_storey = component["total_loss_storey"]
            # Generate a selection of random indices for plotting
            selection = np.random.randint(len(total_loss_storey), size=n_to_plot)
            loss_to_display = {}
            for sel in selection:
                loss_to_display[sel] = total_loss_storey[sel]

            # Plotting the scatter points of the simulations/sampled story losses
            for key in loss_to_display.keys():
                y_scatter = loss_to_display[key] / 10.0 ** 3
                ax.scatter(edp_range, y_scatter, edgecolors=self.color_grid[2], marker='o', s=3,
                           facecolors='none', alpha=0.5)
            # Assigning a label
            ax.scatter(edp_range, y_scatter, edgecolors=self.color_grid[2], marker='o', s=3,
                       facecolors='none', alpha=0.5, label="Simulations")

            # Assignign labels and limits for axes
            if edp in ["IDR", "IDR NS", "IDR S"]:
                xlabel = edp[0:3] + " [%]"
                xlim = [0, 6.]
                ylim = [0, 600.]
            else:
                xlabel = edp[0:3] + " [g]"
                xlim = [0, 4.]
                ylim = [0, 400.]

            ax.set_xlabel(xlabel)
            if id_axis == 0:
                ax.set_ylabel(r"Losses [$10^3 €/100 m^2$]")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.grid(True, which="major", axis="both", ls="--", lw=1.0)
            id_axis += 1

        # Legend
        fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, labels, frameon=False, loc='upper center', fontsize=10,
                   bbox_to_anchor=(0.25, 0., 0.5, -0.15), ncol=5)

        #        fig.tight_layout()

        if not showplot:
            plt.close()
        if sflag:
            self.plot_as_emf(fig, filename=self.directory / "client" / 'figures' / f'loss_all_'
                                                                                   f'{plot_tag}'.replace(" ", "_"))

        # Storing figures
        #        if sflag:
        #            if fig1 is not None:
        #                self.plot_as_emf(fig1, filename=self.directory/"client"/'figures'/f'slf_idr_s_{plot_tag}')
        #                self.plot_as_png(fig1, filename=self.directory/"client"/'figures'/f'slf_idr_s_{plot_tag}')
        #            if fig2 is not None:
        #                self.plot_as_emf(fig2, filename=self.directory/"client"/'figures'/f'slf_pfa_ns_{plot_tag}')
        #                self.plot_as_png(fig2, filename=self.directory/"client"/'figures'/f'slf_pfa_ns_{plot_tag}')
        #            if fig3 is not None:
        #                self.plot_as_emf(fig3, filename=self.directory/"client"/'figures'/f'comp1_frag_{plot_tag}')
        #                self.plot_as_png(fig3, filename=self.directory/"client"/'figures'/f'comp1_frag_{plot_tag}')

        # %%
        # Disaggregating by structural and non-structural contribution
        # Consequence parameters
        #        n1 = 5                                         # Number of structural components
        #        means_cost = np.zeros((n1, 5))
        #
        #        component_data = data["IDR"]["component"].select_dtypes(exclude=["object"])
        #        repair_cost =
        #
        #        # Deriving fragility functions
        #        df = component_data.values[:, 3:]
        #        for item in range(n1):
        #            for ds in range(5):
        #                means_cost[item][ds] = df[item][ds+10]
        #
        #        total_replacement_cost = data[edp]["total_replacement_cost"]
        #        quantities = slf["IDR"]["component"]["Quantity"]
        #        total_s_idr = {}                              # Total structural repair costs
        #        repl_s_idr = {}                               # Replacement costs
        #        for item in range(n1):
        #            total_s_idr[item] = {}
        #            repl_s_idr[item] = max(means_cost[item])
        #            for n in range(3200):
        #                total_repair_cost[item][n] = repair_cost[item][n]*quantities[item-1]

        return data
