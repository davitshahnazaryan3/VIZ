"""
Object for data visualization
"""
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import os
import subprocess


class VIZ:
    def __init__(self):
        """
        Initialize
        """
        self.directory = Path(os.getcwd())
        self.color_grid = ['#840d81','#6c4ba6','#407bc1','#18b5d8','#01e9f5', 
                           '#cef19d','#a6dba7','#77bd98','#398684','#094869']
        
    @staticmethod
    def createFolder(directory):
        """
        Creates a figure if it does not exist
        :param directory: str                       Directory to create
        """
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)
        
    @staticmethod
    def plot_as_emf(figure, **kwargs):
        """
        Saves figure as .emf
        :param figure: fig handle
        :param kwargs: filepath: str                File name, e.g. '*\filename'
        :return: None
        """
        inkscape_path = kwargs.get('inkscape', "C://Program Files//Inkscape//inkscape.exe")
        filepath = kwargs.get('filename', None)
        if filepath is not None:
            path, filename = os.path.split(filepath)
            filename, extension = os.path.splitext(filename)
            svg_filepath = os.path.join(path, filename + '.svg')
            emf_filepath = os.path.join(path, filename + '.emf')
            figure.savefig(svg_filepath, bbox_inches='tight', format='svg')
            subprocess.call([inkscape_path, svg_filepath, '--export-emf', emf_filepath])
            os.remove(svg_filepath)

    @staticmethod
    def plot_as_png(figure, **kwargs):
        """
        Saves figure as .png
        :param figure: fig handle
        :param kwargs: filepath: str                File name, e.g. '*\filename'
        :return: None
        """
        inkscape_path = kwargs.get('inkscape', "C://Program Files//Inkscape//inkscape.exe")
        filepath = kwargs.get('filename', None)
        if filepath is not None:
            path, filename = os.path.split(filepath)
            filename, extension = os.path.splitext(filename)
            svg_filepath = os.path.join(path, filename + '.svg')
            png_filepath = os.path.join(path, filename + '.png')
            figure.savefig(svg_filepath, bbox_inches='tight', format='svg')
            subprocess.call([inkscape_path, svg_filepath, '--export-png', png_filepath])
            os.remove(svg_filepath)
    
    def visualize_slf(self, filename, showplot=False, sflag=False):
        """
        Visualizing graphs for the SLF generator
        :param filename: str                    File name, e.g. '*\filename.extension' 
        :param showplot: bool                   Whether to plot the figures in the interpreter or not
        :param sflag: bool                      Whether to save the figures or not
        """
        filepath = self.directory / "client" / filename
        
        if filename.endswith(".pkl") or filename.endswith(".pickle"):
            with open(filepath, 'rb') as file:
                data = pickle.load(file)
        else:
            raise ValueError("[EXCEPTION] Currently only pickle or pkl file format is accepted")
          
        idr_range = data["IDR NS"]["fragilities"]["EDP"]
        pfa_range = data["PFA NS"]["fragilities"]["EDP"]
        
        # IDR sensitive structural and non-structural performance group SLFs
        fig1, ax = plt.subplots(figsize=(4, 3), dpi=100)
        cnt = 0
        for key in data["SLFs"]:
            y = data["SLFs"][key]
            if "IDR" in key:
                plt.plot(idr_range, y, color=self.color_grid[cnt], label=key)
            cnt += 1
        plt.xlabel('IDR')
        plt.ylabel('E(L | IDR)')
        plt.xlim(0, 0.2)
        plt.ylim(0, 1)
        plt.grid(True, which="major", axis="both", ls="--", lw=1.0)
        plt.legend(frameon=False, loc="upper right", fontsize=10)
        if not showplot:
            plt.close()
        
        # PFA sensitive non-structural performance group SLFs
        fig2, ax = plt.subplots(figsize=(4, 3), dpi=100)
        cnt = 0
        for key in data["SLFs"]:
            y = data["SLFs"][key]
            if "PFA" in key:
                plt.plot(pfa_range, y, color=self.color_grid[cnt], label=key)
            cnt += 1
        plt.xlabel('PFA [g]')
        plt.ylabel('E(L | IDR)')
        plt.xlim(0, 10)
        plt.ylim(0, 1)
        plt.grid(True, which="major", axis="both", ls="--", lw=1.0)
        plt.legend(frameon=False, loc="upper right", fontsize=10)
        if not showplot:
            plt.close()
            
        # Sample fragility function
        component = data["IDR S"]["fragilities"]["ITEMs"][1]
        fig3, ax = plt.subplots(figsize=(4, 3), dpi=100)
        cnt = 0
        for key in component.keys():
            plt.plot(idr_range, component[key], color=self.color_grid[cnt], label=key)
            cnt += 1
        plt.xlabel('IDR')
        plt.ylabel('Probability of exceeding DS')
        plt.xlim(0, 0.2)
        plt.ylim(0, 1)
        plt.grid(True, which="major", axis="both", ls="--", lw=1.0)
        plt.legend(frameon=False, loc="upper right", fontsize=10)
        if not showplot:
            plt.close()
        
        # EDP vs loss ratio, Drift-sensitive Structural components
        fig4, ax = plt.subplots(figsize=(4, 3), dpi=100)
        cnt = 0
        for key in data["IDR S"]["edp_dv_fitted"].keys():
            y = data["IDR S"]["edp_dv_fitted"][key]
            plt.plot(idr_range, y, color=self.color_grid[cnt], label=key)
            cnt += 2
        plt.xlabel("IDR")
        plt.ylabel("Loss Ratio")
        plt.xlim(0, 0.2)
        plt.ylim(0, 1.4)
        plt.grid(True, which="major", axis="both", ls="--", lw=1.0)
        plt.legend(frameon=False, loc="best", fontsize=10)
        if not showplot:
            plt.close()
        
        # EDP vs loss ratio, Drift-sensitive Non-structural components
        fig5, ax = plt.subplots(figsize=(4, 3), dpi=100)
        cnt = 0
        for key in data["IDR NS"]["edp_dv_fitted"].keys():
            y = data["IDR NS"]["edp_dv_fitted"][key]
            plt.plot(idr_range, y, color=self.color_grid[cnt], label=key)
            cnt += 2
        plt.xlabel("IDR")
        plt.ylabel("Loss Ratio")
        plt.xlim(0, 0.2)
        plt.ylim(0, 1.4)
        plt.grid(True, which="major", axis="both", ls="--", lw=1.0)
        plt.legend(frameon=False, loc="best", fontsize=10)
        if not showplot:
            plt.close()
            
        # EDP vs loss ratio, Acceleration-sensitive Non-structural components
        fig6, ax = plt.subplots(figsize=(4, 3), dpi=100)
        cnt = 0
        for key in data["PFA NS"]["edp_dv_fitted"].keys():
            y = data["PFA NS"]["edp_dv_fitted"][key]
            plt.plot(pfa_range, y, color=self.color_grid[cnt], label=key)
            cnt += 2
        plt.xlabel("PFA [g]")
        plt.ylabel("Loss Ratio")
        plt.xlim(0, 10.0)
        plt.ylim(0, 1.4)
        plt.grid(True, which="major", axis="both", ls="--", lw=1.0)
        plt.legend(frameon=False, loc="best", fontsize=10)
        if not showplot:
            plt.close()
            
        # Loss in euro vs EDP (including the simulation scatter, the fractiles of 
        #  the simulations and the fitted fractiles, and the fitted mean)
        # IDR-S
        fig7, ax = plt.subplots(figsize=(4, 3), dpi=100)
        component = data["IDR S"]
        idr_range = idr_range*100
        cnt = 0
        for key in component["edp_dv_euro"].keys():
            y_fit = component["edp_dv_euro"][key] / 10.0**3
            y = component["losses"]["loss_curve"].loc[key] / 10.0**3
            plt.plot(idr_range, y, color=self.color_grid[cnt], label=key, alpha=0.5, marker='o', markersize=3)
            plt.plot(idr_range, y_fit, color=self.color_grid[cnt], label=key)
            cnt +=2
        total_loss_storey = component["total_loss_storey"]
        for key in total_loss_storey.keys():
            y_scatter = total_loss_storey[key] / 10.0**3
            plt.scatter(idr_range, y_scatter, edgecolors=self.color_grid[2], marker='o', s=3,
                        facecolors='none', alpha=0.5)
        plt.scatter(idr_range, y_scatter, edgecolors=self.color_grid[2], marker='o', s=3,
                    facecolors='none', alpha=0.5, label="Simulations")
        plt.xlabel("IDR [%]")
        plt.ylabel(r"Losses [$10^3 €/100 m^2$]")
        plt.xlim(0, 15)
        plt.ylim(0, 250)
        plt.grid(True, which="major", axis="both", ls="--", lw=1.0)
        ax.legend(frameon=False, loc='center left', fontsize=10, bbox_to_anchor=(1, 0.5))
        if not showplot:
            plt.close()
            
        # IDR-NS
        fig8, ax = plt.subplots(figsize=(4, 3), dpi=100)
        component = data["IDR NS"]
        cnt = 0
        for key in component["edp_dv_euro"].keys():
            y_fit = component["edp_dv_euro"][key] / 10.0**3
            y = component["losses"]["loss_curve"].loc[key] / 10.0**3
            plt.plot(idr_range, y, color=self.color_grid[cnt], label=key, alpha=0.5, marker='o', markersize=3)
            plt.plot(idr_range, y_fit, color=self.color_grid[cnt], label=key)
            cnt +=2
        total_loss_storey = component["total_loss_storey"]
        for key in total_loss_storey.keys():
            y_scatter = total_loss_storey[key] / 10.0**3
            plt.scatter(idr_range, y_scatter, edgecolors=self.color_grid[2], marker='o', s=3,
                        facecolors='none', alpha=0.5)
        plt.scatter(idr_range, y_scatter, edgecolors=self.color_grid[2], marker='o', s=3,
                    facecolors='none', alpha=0.5, label="Simulations")
        plt.xlabel("IDR [%]")
        plt.ylabel(r"Losses [$10^3 €/100 m^2$]")
        plt.xlim(0, 15)
        plt.ylim(0, 50)
        plt.grid(True, which="major", axis="both", ls="--", lw=1.0)
        ax.legend(frameon=False, loc='center left', fontsize=10, bbox_to_anchor=(1, 0.5))
        if not showplot:
            plt.close()
            
        # PFA-NS
        fig9, ax = plt.subplots(figsize=(4, 3), dpi=100)
        component = data["PFA NS"]
        cnt = 0
        for key in component["edp_dv_euro"].keys():
            y_fit = component["edp_dv_euro"][key] / 10.0**3
            y = component["losses"]["loss_curve"].loc[key] / 10.0**3
            plt.plot(pfa_range, y, color=self.color_grid[cnt], label=key, alpha=0.5, marker='o', markersize=3)
            plt.plot(pfa_range, y_fit, color=self.color_grid[cnt], label=key)
            cnt +=2
        total_loss_storey = component["total_loss_storey"]
        for key in total_loss_storey.keys():
            y_scatter = total_loss_storey[key] / 10.0**3
            plt.scatter(pfa_range, y_scatter, edgecolors=self.color_grid[2], marker='o', s=3,
                        facecolors='none', alpha=0.5)
        plt.scatter(pfa_range, y_scatter, edgecolors=self.color_grid[2], marker='o', s=3,
                    facecolors='none', alpha=0.5, label="Simulations")
        plt.xlabel("PFA [g]")
        plt.ylabel(r"Losses [$10^3 €/100 m^2$]")
        plt.xlim(0, 10)
        plt.ylim(0, 50)
        plt.grid(True, which="major", axis="both", ls="--", lw=1.0)
        ax.legend(frameon=False, loc='center left', fontsize=10, bbox_to_anchor=(1, 0.5))
        if not showplot:
            plt.close()
            
        # Storing figures
        if sflag:
            self.plot_as_emf(fig1,filename=self.directory/"client"/'figures'/'slf_idr_s')
            self.plot_as_png(fig1,filename=self.directory/"client"/'figures'/'slf_idr_s')
            self.plot_as_emf(fig2,filename=self.directory/"client"/'figures'/'slf_idr_ns')
            self.plot_as_png(fig2,filename=self.directory/"client"/'figures'/'slf_idr_ns')
            self.plot_as_emf(fig3,filename=self.directory/"client"/'figures'/'slf_pfa_ns')
            self.plot_as_png(fig3,filename=self.directory/"client"/'figures'/'slf_pfa_ns')
            self.plot_as_emf(fig4,filename=self.directory/"client"/'figures'/'edp_loss_idr_s')
            self.plot_as_png(fig4,filename=self.directory/"client"/'figures'/'edp_loss_idr_s')
            self.plot_as_emf(fig5,filename=self.directory/"client"/'figures'/'edp_loss_idr_ns')
            self.plot_as_png(fig5,filename=self.directory/"client"/'figures'/'edp_loss_idr_ns')
            self.plot_as_emf(fig6,filename=self.directory/"client"/'figures'/'edp_loss_pfa_ns')
            self.plot_as_png(fig6,filename=self.directory/"client"/'figures'/'edp_loss_pfa_ns')
            self.plot_as_emf(fig7,filename=self.directory/"client"/'figures'/'loss_idr_s')
            self.plot_as_png(fig7,filename=self.directory/"client"/'figures'/'loss_idr_s')
            self.plot_as_emf(fig8,filename=self.directory/"client"/'figures'/'loss_idr_ns')
            self.plot_as_png(fig8,filename=self.directory/"client"/'figures'/'loss_idr_ns')
            self.plot_as_emf(fig9,filename=self.directory/"client"/'figures'/'loss_pfa_ns')
            self.plot_as_png(fig9,filename=self.directory/"client"/'figures'/'loss_pfa_ns')
        
        return data
    

if __name__ == "__main__":
    viz = VIZ()
    viz.createFolder(viz.directory/"client"/"figures")
    slf = viz.visualize_slf("slf1.pkl", showplot=True, sflag=False)


























