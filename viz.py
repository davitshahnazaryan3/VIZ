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

    """Most of the input manipulation and method calling happens here for now, until a GUI 
    (even a simple one) is created"""
    # Calling the master object
    exportDir = Path.cwd().parents[0] / ".applications/case1/figures"
    viz = VIZ(exportDir)

    # Call the earthquake engineering master file
    seismic = SeismicMaster(export=False, exportDir=exportDir)

    # # Hazard plotting
    # pathHazard = Path.cwd().parents[0] / ".applications/case1/Hazard-LAquila-Soil-C.pkl"
    # pathFitted = Path.cwd().parents[0] / ".applications/case1/Output1/fit_Hazard-LAquila-Soil-C.pkl"
    # pathCoefs = Path.cwd().parents[0] / ".applications/case1/Output1/coef_Hazard-LAquila-Soil-C.pkl"
    # period = [1.0, 0.]
    # rp = [475, 10000]
    # seismic.hazard(pathHazard, pathFitted=pathFitted, pathCoefs=pathCoefs, period=period, rp=rp,
    #                true=False, fitted=False, both=True)

    # # SLF plotting
    path = Path.cwd().parents[0] / ".applications/case1/Output/slfoutput"
    nst = 5
    seismic.slfs(path, nst=nst, detailedSLF=False)

    # SPO plotting
    # path = Path.cwd().parents[0] / ".applications/case1/Output1/RCMRF/SPO.pickle"
    # seismic.spo(path)
    
    # # IPBSD plotting
    # lossCurvePath = Path.cwd().parents[0] / ".applications/case1/Output/Cache/lossCurve.pickle"
    # spectrumPath = Path.cwd().parents[0] / ".applications/case1/Output/Cache/sls_spectrum.csv"
    # solutionPath = Path.cwd().parents[0] / ".applications/case1/Output/Cache/ipbsd.pickle"
    # spo2idaPath = Path.cwd().parents[0] / ".applications/case1/Output/Cache/spoAnalysisCurveShape.pickle"
    # spoModelPath = Path.cwd().parents[0] / ".applications/case1/Output/Cache/modelOutputs.pickle"
    
    # seismic.ipbsd(lossCurvePath=lossCurvePath, spectrumPath=spectrumPath, solutionPath=solutionPath,
    #               spo2idaPath=spo2idaPath, spoModelPath=spoModelPath)

    # RCMRF IDA plotting
    # ida_rcmrfPath = Path.cwd().parents[0] / ".applications/case1/Output1/RCMRF/ida_cache.pickle"
    # spoPath = Path.cwd().parents[0] / ".applications/case1/Output1/RCMRF/SPO.pickle"
    # ipbsdPath = Path.cwd().parents[0] / ".applications/case1/Output1/Cache/ipbsd.pickle"
    # spo2idaPath = Path.cwd().parents[0] / ".applications/case1/Output1/Cache/spoAnalysisCurveShape.pickle"
    # seismic.rcmrf(ida_rcmrfPath=ida_rcmrfPath, spoModelPath=spoPath, ipbsdPath=ipbsdPath, spo2idaPath=spo2idaPath)
