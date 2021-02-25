"""
Object for data visualization
"""
from pathlib import Path
import os
from EqEngineering.seismicMaster import SeismicMaster


class VIZ:
    def __init__(self, exportDir):
        """
        :param exportDir: str               Export directory
        """
        self.exportDir = exportDir
        self.create_folder(self.exportDir)

    @staticmethod
    def create_folder(directory):
        """
        Creates a figure if it does not exist
        :param directory: str                       Directory to create
        """
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)


if __name__ == "__main__":

    # TODO, at the moment pflag is gatekeeping export of results as well, modify to trigger only plotting
    """Most of the input manipulation and method calling happens here for now, until a GUI 
    (even a simple one) is created"""
    # Calling the master object
    case_dir = Path.cwd().parents[0] / ".applications/LOSS Validation Manuscript/Case2"
    exportDir = case_dir / "figures"
    viz = VIZ(exportDir)
    
    # Initialize
    lossCurvePath = None
    spectrumPath = None
    solutionPath = None
    spo2idaPath = None
    spoModelPath = None
    
    # Call the earthquake engineering master file
    seismic = SeismicMaster(export=True, exportDir=exportDir)

    # # Hazard plotting
    # pathHazard = case_dir / "Hazard-LAquila-Soil-C.pkl"
    # pathFitted = case_dir / "fit_Hazard-LAquila-Soil-C.pkl"
    # pathCoefs = case_dir / "coef_Hazard-LAquila-Soil-C.pkl"
    # period = [0.6, 0.72, 0.]
    # rp = [475, 10000]
    # seismic.hazard(pathHazard, pathFitted=pathFitted, pathCoefs=pathCoefs, period=period, rp=rp,
    #                true=False, fitted=False, both=True)

    # # SLF plotting
    # path = Path.cwd().parents[0] / ".applications/LOSS Validation Manuscript/Case2/slfoutput"
    # viz.create_folder(path)
    # nst = 4
    # seismic.slfs(path, nst=nst, detailedSLF=False)

    # # SPO plotting
    # path = case_dir / "RCMRF/SPO.pickle"
    # seismic.spo(path)
    
    # IPBSD plotting
    # lossCurvePath = case_dir / "Cache/framex/lossCurve.pickle"
    # spectrumPath = case_dir / "Cache/sls_spectrum.csv"
    solutionPath = case_dir / "Cache/framey/ipbsd.pickle"
    # spo2idaPath = case_dir / "Cache/framey/spoAnalysisCurveShape.pickle"
    # spoModelPath = case_dir / "Cache/framey/modelOutputs.pickle"
    
    seismic.ipbsd(lossCurvePath=lossCurvePath, spectrumPath=spectrumPath, solutionPath=solutionPath,
                  spo2idaPath=spo2idaPath, spoModelPath=spoModelPath)
    
    # RCMRF IDA plotting
    # ida_rcmrfPath = Path.cwd().parents[0] / ".applications/case1/Output1/RCMRF/ida_cache.pickle"
    # spoPath = Path.cwd().parents[0] / ".applications/case1/Output1/RCMRF/SPO.pickle"
    # ipbsdPath = Path.cwd().parents[0] / ".applications/case1/Output1/Cache/ipbsd.pickle"
    # spo2idaPath = Path.cwd().parents[0] / ".applications/case1/Output1/Cache/spoAnalysisCurveShape.pickle"
    # seismic.rcmrf(ida_rcmrfPath=ida_rcmrfPath, spoModelPath=spoPath, ipbsdPath=ipbsdPath, spo2idaPath=spo2idaPath)
