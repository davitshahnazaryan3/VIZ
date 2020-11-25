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
    # period = 1.05
    # seismic.hazard(pathHazard, period=period)

    # # SLF plotting
    # path = Path.cwd().parents[0] / ".applications/case1/Output/slfoutput"
    # nst = 5
    # seismic.slfs(path, nst=nst)

    # SPO plotting
    path = Path.cwd().parents[0] / ".applications/case1/Output/RCMRF/SPO.pickle"
    seismic.spo(path)
    