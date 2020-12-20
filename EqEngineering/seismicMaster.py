from EqEngineering.hazard import Hazard
from EqEngineering.slf import SLF
from EqEngineering.ipbsd import IPBSD
from EqEngineering.spo import SPO
from EqEngineering.ida import IDA
from EqEngineering.loss import Loss
import pickle
from plotter import Plotter
import pandas as pd


class SeismicMaster:
    def __init__(self, export=False, exportMiser=False, exportDir=None):
        """
        Initialize visualization
        :param export: bool                 Export figures or not?
                                            (exports in .emf via inkscape, modify function if need be)
        :param exportMiser: bool            Export png images or not? (It's a trap...)
        :param exportDir: str               Export directory
        """
        self.export = export
        self.exportMiser = exportMiser
        self.exportDir = exportDir
        if export or exportMiser:
            self.plotter = Plotter()

    def exportFigure(self, fig, name):
        """
        Exports figures if necessary
        :param fig: figure object
        :param name: str                    Base name of the file
        :return: None
        """
        if self.export:
            self.plotter.plot_as_emf(fig, filename=self.exportDir / name)
        if self.exportMiser:
            self.plotter.plot_as_png(fig, filename=self.exportDir / name)

    def hazard(self, path, true=True, pathFitted=None, pathCoefs=None, fitted=False, period=None, rp=None, both=False):
        """
        Calls a Hazard object
        :param path: str                    Path to file
        :param true: bool                   Plot true hazard function
        :param pathFitted: str              Path to fitted function
        :param pathCoefs: str               Path to fitted function coefficients
        :param fitted: bool                 Plot 2-nd order fitted function
        :param period: float                Period of interest to highlight
        :param rp: list(float)              Return periods to highlight on the graphs
        :param both: bool                   Plots seismic hazard of true and fitted functions
        :return: None
        """
        # Call the Hazard object
        hazard = Hazard(period, rp=rp)
        if true:
            with open(path, "rb") as f:
                h = pickle.load(f)
            fig = hazard.true_hazard(h)
            self.exportFigure(fig, "trueHazard")

        if fitted:
            with open(pathFitted, "rb") as f:
                h = pickle.load(f)
            with open(pathCoefs, "rb") as f:
                coefs = pickle.load(f)
            fig = hazard.fitted_hazard(h, coefs)
            self.exportFigure(fig, "fittedHazard")

        if both:
            with open(path, "rb") as f:
                h = pickle.load(f)
            with open(pathFitted, "rb") as f:
                fitted = pickle.load(f)
            with open(pathCoefs, "rb") as f:
                coefs = pickle.load(f)
            fig = hazard.both_hazard(h, fitted, coefs)
            self.exportFigure(fig, "Hazard")

        if self.export or self.exportMiser:
            print("[SUCCESS] Hazard functions have been exported!")
        else:
            print("[SUCCESS] Hazard functions have been plotted!")

    def slfs(self, path, nst, n_to_plot=100, geometry=0, normalizeCost=1, pflag=True, detailedSLF=False):
        """
        Plots graphs separately for each storey and Performance group (i.e. 3 groups x n storeys)
        :param path: str                    Directory of SLF output files
        :param nst: int
        :param n_to_plot: int               Number of scatter points to plot
        :param geometry: int
        :param normalizeCost: float
        :param pflag: bool
        :param detailedSLF: bool
        :return: None
        """
        slf = SLF(path, nst, n_to_plot, geometry, normalizeCost)
        if pflag:
            loss = None
            
            if detailedSLF:
                figs, names = slf.slfs_detailed()
            else:
                figs, names, loss, edps = slf.read_slfs()

            for i in range(len(figs)):
                self.exportFigure(figs[i], names[i])
                if loss is not None:
                    with open(self.exportDir / "SLF_losses.pickle", "wb") as handle:
                        pickle.dump(loss, handle)
                    with open(self.exportDir / "SLF_edps.pickle", "wb") as handle:
                        pickle.dump(edps, handle)

        if self.export or self.exportMiser:
            print("[SUCCESS] Storey loss functions have been exported!")
        else:
            print("[SUCCESS] Storey loss functions have been plotted!")

    def spo(self, path, pflag=True):
        """
        Plots SPO
        :param path: str
        :param pflag: bool
        :return: None
        """
        spo = SPO()
        if pflag:
            with open(path, "rb") as f:
                data = pickle.load(f)
            fig = spo.base_shear_vs_top_displacement(data)
            self.exportFigure(fig, "rcmrf_spo")

        if self.export or self.exportMiser:
            print("[SUCCESS] SPO figure has been exported!")
        else:
            print("[SUCCESS] SPO figure has been plotted!")

    def ipbsd(self, lossCurvePath=None, spectrumPath=None, solutionPath=None, spo2idaPath=None, spoModelPath=None):
        """
        IPBSD plotting
        :param lossCurvePath: str                   Loss curve path
        :param spectrumPath: str                    Spectrum path
        :param solutionPath: str
        :param spo2idaPath: str
        :param spoModelPath: str
        :param ida_rcmrfPath: str
        :return:
        """
        ipbsd = IPBSD()
        # Loss curve plotter
        if lossCurvePath is not None:
            with open(lossCurvePath, "rb") as f:
                data = pickle.load(f)
            fig = ipbsd.lossCurve(data)
            self.exportFigure(fig, "LossCurve")

        # Spectrum plotter
        if spectrumPath is not None:
            spectrum = pd.read_csv(spectrumPath)
            fig = ipbsd.spectrum(spectrum, x="Sd", y="Sa")
            self.exportFigure(fig, "spectrum")

            # Design solution space plotter
            if solutionPath is not None:
                with open(solutionPath, "rb") as f:
                    sol = pickle.load(f)
                fig = ipbsd.solutionSpace(sol, spectrum)
                self.exportFigure(fig, "solution_space")

        # SPO2IDA results plotter
        if spo2idaPath is not None:
            with open(spo2idaPath, "rb") as f:
                spo2ida = pickle.load(f)
            fig = ipbsd.spo2ida(spo2ida)
            self.exportFigure(fig, "spo2ida")

        # SPO shape from a nonlinear model and its idealized shape
        if spoModelPath is not None:
            with open(spoModelPath, "rb") as f:
                spo = pickle.load(f)
            fig = ipbsd.spo(spo)
            self.exportFigure(fig, "spo")

        # --------------------------------------------------------------------
        # -----------------    Printing stuff --------------------------------
        if lossCurvePath is not None:
            if self.export or self.exportMiser:
                print("[SUCCESS] Loss Curve figure has been exported!")
            else:
                print("[SUCCESS] Loss Curve figure has been plotted!")
        if spectrumPath is not None:
            if self.export or self.exportMiser:
                print("[SUCCESS] Spectrum figure has been exported!")
            else:
                print("[SUCCESS] Spectrum figure has been plotted!")
            if solutionPath is not None:
                if self.export or self.exportMiser:
                    print("[SUCCESS] Design solution space figure has been exported!")
                else:
                    print("[SUCCESS] Design solution space figure has been plotted!")

        if spo2idaPath is not None:
            if self.export or self.exportMiser:
                print("[SUCCESS] SPO2IDA figure has been exported!")
            else:
                print("[SUCCESS] SPO2IDA figure has been plotted!")

        if spoModelPath is not None:
            if self.export or self.exportMiser:
                print("[SUCCESS] Model SPO figure has been exported!")
            else:
                print("[SUCCESS] Model SPO figure has been plotted!")

    def rcmrf(self, ida_rcmrfPath=None, spoModelPath=None, ipbsdPath=None, spo2idaPath=None):
        """
        Plotting RCMRF results
        :param ida_rcmrfPath: str
        :param spoModelPath: str
        :param ipbsdPath: str
        :param spo2idaPath: str
        :return:
        """
        ida = IDA()
        # RCMRF IDA plotter
        if ida_rcmrfPath is not None:
            with open(ida_rcmrfPath, "rb") as f:
                ida_cache = pickle.load(f)
            fig = ida.disp_vs_im(ida_cache)
            self.exportFigure(fig, "ida_rcmrf_disp_im")

            # Comparing SPO2IDA and Model IDA results
            if spoModelPath is not None:
                with open(spoModelPath, "rb") as f:
                    spo = pickle.load(f)
                with open(ipbsdPath, "rb") as f:
                    ipbsd = pickle.load(f)
                with open(spo2idaPath, "rb") as f:
                    spo2ida = pickle.load(f)
                fig = ida.spo2ida_model(ida_cache, spo, ipbsd, spo2ida)
                self.exportFigure(fig, "ida_spo2ida")
