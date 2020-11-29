"""
Postprocessing of results
"""
import pickle
import numpy as np
import pandas as pd
import math
import json
from scipy.interpolate import interp1d


class Postprocessor:
    def __init__(self, path, export=True):
        """
        :param path: str                    IDA results directory"""
        self.path = path
        self.export = export

    def export_results(self, filepath, data, filetype):
        """
        Store results in the database
        :param filepath: str                            Filepath, e.g. "directory/name"
        :param data:                                    Data to be stored
        :param filetype: str                            Filetype, e.g. npy, json, pkl, csv
        :return: None
        """
        if filetype == "npy":
            np.save(f"{filepath}.npy", data)
        elif filetype == "pkl" or filetype == "pickle":
            with open(f"{filepath}.pickle", 'wb') as handle:
                pickle.dump(data, handle)
        elif filetype == "json":
            with open(f"{filepath}.json", "w") as json_file:
                json.dump(data, json_file)
        elif filetype == "csv":
            data.to_csv(f"{filepath}.csv", index=False)

    def cscvn2(self, points):
        cs = interp1d(points[0], points[1])
        return cs

    def splinequery_IDA(self, spl, edp_range, qtile_range, edp, num_recs):

        # Getting the edp-based im values
        im_spl = np.zeros([num_recs, len(edp_range)])
        for j in range(num_recs):
            for i in range(len(edp_range)):
                if edp_range[i] <= max(edp[j]):
                    im_spl[j][i] = spl[j](edp_range[i])
                else:
                    im_spl[j][i] = im_spl[j][i - 1]

        # Getting the edp-based im quantiles
        im_qtile = np.zeros([len(qtile_range), len(edp_range)])
        for q in range(len(qtile_range)):
            for i in range(len(edp_range)):
                im_qtile[q][i] = np.quantile(im_spl[:, i], qtile_range[q])

        return im_spl, im_qtile

    def splinefit_IDA(self, im, edp, nrecs):

        # Fit the spline and get the values
        spl = {}
        for i in range(nrecs):
            im[i] = np.sort(im[i])
            ind = np.argsort(im[i])
            edp[i] = edp[i][ind]
            ind_mx = int(np.where(edp[i] == max(edp[i]))[0])
            temp_im = im[i]
            temp_edp = edp[i]
            #        if ind_mx < 15:
            #            temp_im = np.delete(im[i],np.arange(ind_mx+1,len(im[i])))
            #            temp_edp = np.delete(edp[i],np.arange(ind_mx+1,len(edp[i])))
            spl[i] = self.cscvn2([temp_edp, temp_im])

        del ind, temp_im, temp_edp

        return spl

    def ida(self, IMpath, dursPath, res_time=10.):
        """
        Postprocess IDA results
        :param IMpath: str                  Path to IM file
        :param dursPath: str                Path to the file containing durations of each record
        :param res_time: float              Free vibrations time added to the end of the record
        :return:
        """
        '''
        Current structure of the single pickle file (a very large file I must say )
        Record -> Runs -> EDP type (0=PFA, 1=Displacement, 2=PSD) -> 
        -> array of shape [n x Record length]
        where n is nst for PSD, and nst + 1 for PFA and Displacement
        '''
        """
        The postprocessed file will have the following structure (looks a bit uncomfy though): 
        1. IDA; 2. summary_results (2 keys)
        1.1 ground motions (n_rec keys) -> IM, ISDR, PFA, RISDR (4 keys) -> each being a list of size of number of runs
        2.1 ground motions (n_rec keys) -> IM levels (n_runs keys) -> maxFA, maxISDR (2 keys) -> 
        -> number of storeys (nst for maxISDR) /floors (nst+1 for maxFA) keys -> a single value
        """
        # Read the IDA outputs
        with open(self.path, 'rb') as file:
            data = pickle.load(file)

        # Read the IDA IM levels
        IM = np.genfromtxt(IMpath, delimiter=',')

        # Read the durations of the records
        durs = list(pd.read_csv(dursPath, header=None)[0])

        # Number of records
        nrecs = len(data)
        # Number of runs per each record
        nruns = len(data[list(data.keys())[0]])

        # Initialize some variables
        im = np.zeros([nrecs, nruns + 1])
        idx = np.zeros([nrecs, nruns], dtype='i')
        mpfa_us = np.full([nrecs, nruns], np.nan)
        mpsd_us = np.full([nrecs, nruns], np.nan)
        mrpsd_us = np.full([nrecs, nruns], np.nan)
        mtdisp_us = np.full([nrecs, nruns + 1], np.nan)
        mtrx = np.full([nrecs, nruns + 1], np.nan)
        mpfa = np.zeros([nrecs, nruns + 1])
        mpsd = np.zeros([nrecs, nruns + 1])
        mtdisp = np.zeros([nrecs, nruns + 1])

        # Initialize target dictionary with its first stage
        res = {'IDA': {}, 'summary_results': {}}
        resKeys = list(res.keys())

        # Loop for each record
        for rec in range(1, nrecs+1):
            print("gm_%s" % rec)

            # Second stage of the dictionary
            res[resKeys[0]][rec] = {'IM': [], 'ISDR': [], 'PFA': [], 'RISDR': []}
            res[resKeys[1]][rec] = {}

            # Add IM values into the results file
            res[resKeys[0]][rec]["IM"] = IM[rec - 1]

            # Sort the IM values
            im[rec - 1, 1:] = np.sort(IM[rec - 1])
            idx[rec - 1, :] = np.argsort(IM[rec - 1])

            # Third stage of the dictionary
            for i in im[rec - 1, 1:]:
                i = str(np.round(i, 2))
                res[resKeys[1]][rec][i] = {'maxFA': {}, 'maxISDR': {}, 'maxRISDR': {}}

            # Loop over each run
            for run in range(1, nruns + 1):
                # Select analysis results of rec and run
                selection = data[rec - 1][run]

                # Get PFAs in g
                pfa = np.amax(abs(selection[0][:, 1:]), axis=1)

                # IML in g
                iml = str(np.round(IM[rec - 1][run - 1], 2))

                for st in range(len(pfa)):
                    res[resKeys[1]][rec][iml]["maxFA"][st] = pfa[st]
                mpfa_us[rec - 1, run - 1] = max(pfa)

                # Get PSDs in %
                psd = np.amax(abs(selection[2]), axis=1)

                for st in range(len(psd)):
                    res[resKeys[1]][rec][iml]["maxISDR"][st + 1] = psd[st]
                mpsd_us[rec - 1, run - 1] = max(psd)

                # Getting the residual PSDs
                # Analysis time step
                dt = (durs[rec - 1] + res_time) / selection[0].shape[1]
                idxres = int((durs[rec - 1] - res_time) / dt)

                resDrifts = selection[2][:, idxres:]
                for st in range(len(psd)):
                    res[resKeys[1]][rec][iml]["maxRISDR"][st + 1] = sum(resDrifts[st]) / len(resDrifts[st])
                # Record the peak value of residual drift at each run for each record
                mrpsd_us[rec - 1, run - 1] = max(np.sum(resDrifts, axis=1) / resDrifts.shape[1])

                # Get the top displacement in m
                top_disp = np.amax(abs(selection[1]), axis=1)
                mtdisp_us[rec - 1, run - 1] = top_disp[-1]

                # Sort the results
                res["IDA"][rec]["PFA"] = mpfa_us[run - 1, :]
                res["IDA"][rec]["ISDR"] = mpsd_us[run - 1, :]
                res["IDA"][rec]["RISDR"] = mrpsd_us[run - 1, :]

                # Repopulate nans with max of data
                # res["IDA"][rec]["RISDR"] = [max(res['IDA'][rec]['RISDR']) if math.isnan(x) else x for
                # x in res['IDA'][rec]['RISDR']]

                mpfa[rec - 1, 1:] = mpfa_us[rec - 1, :][idx[rec - 1]]
                mpsd[rec - 1, 1:] = mpsd_us[rec - 1, :][idx[rec - 1]]
                mtdisp[rec - 1, 1:] = mtdisp_us[rec - 1, :][idx[rec - 1]]

        # Fit the splines to the data
        mpsd_range = np.linspace(0.01, 20, 200)
        mtdisp_range = np.linspace(0.01, 1, 200)
        mpfa_range = np.linspace(0.01, 3, 200)
        # Quantile ranges to visualize for the IDAs
        qtile_range = np.array([0.16, 0.5, 0.84])

        spl_mtdisp = self.splinefit_IDA(im, mtdisp, nrecs)
        spl_mpfa = self.splinefit_IDA(im, mpfa, nrecs)
        spl_mpsd = self.splinefit_IDA(im, mpsd, nrecs)

        im_spl, im_qtile = self.splinequery_IDA(spl_mtdisp, mtdisp_range, qtile_range, mtdisp, nrecs)
        mpfa_spl, mpfa_qtile = self.splinequery_IDA(spl_mpfa, mpfa_range, qtile_range, mpfa, nrecs)
        mpsd_spl, mpsd_qtile = self.splinequery_IDA(spl_mpsd, mpsd_range, qtile_range, mpsd, nrecs)

        # Creating a dictionary for the spline fits
        cache = {"im_spl": im_spl, "im_qtile": im_qtile, "mtdisp": mtdisp_range}

        # Exporting
        if self.export:
            self.export_results(self.path.parents[0] / "ida_processed", res, "pickle")
            self.export_results(self.path.parents[0] / "ida_cache", cache, "pickle")
            print("[SUCCESS] Postprocesssing complete. Results have been exported!")
        else:
            print("[SUCCESS] Postprocesssing complete.")

        return res


if __name__ == "__main__":

    from pathlib import Path
    directory = Path.cwd()

    path = directory.parents[0] / ".applications/case1/Output/RCMRF/IDA.pickle"
    IMpath = directory.parents[0] / ".applications/case1/Output/RCMRF/IM.csv"
    dursPath = directory.parents[0] / "RCMRF/sample/groundMotion/GMR_durs.txt"
    export = True

    p = Postprocessor(path, export=export)
    results = p.ida(IMpath, dursPath)
