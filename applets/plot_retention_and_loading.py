"""
arguments are counts, counts2, and a defined threshold to discriminate between atom
and background. todo: make the threshold parameter optional and instead determine it
with a fit

applet command:
python "C:\..\qn_artiq_routines\applets\plot_retention_and_loading.py"
photocounts photocounts2 n_measurements iteration single_atom_counts_per_s t_SPCM_first_shot
"""

#!/usr/bin/env python3

import numpy as np
import PyQt5  # make sure pyqtgraph imports Qt5
from PyQt5.QtCore import QTimer
import pyqtgraph
from skimage.filters import threshold_otsu
from artiq.applets.simple import TitleApplet


class XYPlot(pyqtgraph.PlotWidget):
    def __init__(self, args):
        pyqtgraph.PlotWidget.__init__(self)
        self.args = args
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        # self.timer.timeout.connect(self.length_warning)
        self.mismatch = {'X values': False,
                         'Error bars': False,
                         'Fit values': False}

    def data_changed(self, data, mods, title):

        try: # not all of these are persistent
            counts_shot1 = np.array(data[self.args.counts_shot1][1][1:])
            counts_shot2 = np.array(data[self.args.counts_shot2][1][1:])
            measurements = data[self.args.measurements][1]
            threshold_cts_per_s = data[self.args.threshold_cts_per_s][1]
            t_exposure = data[self.args.t_exposure][1]
            cutoff = int(t_exposure*threshold_cts_per_s)

            iteration = len(counts_shot1)//measurements
            if iteration > 0:
                if len(counts_shot1) == len(counts_shot2):

                    retention_array = np.zeros(iteration)
                    loading_rate_array = np.zeros(iteration)
                    n_atoms_loaded_array = np.zeros(iteration)

                    x = np.arange(iteration)

                    try:
                        nsteps = len(data.get(self.args.scan_sequence1, (False, None))[1])
                        scan_sequence1 = data[self.args.scan_sequence1][1]
                        if nsteps > 1 or scan_sequence1 != [0.0]:
                            x = np.array(scan_sequence1[:iteration])
                    except: # len will fail if sequence is None
                        pass

                    # if iteration > 0:
                    for i in range(iteration):
                        shot1 = counts_shot1[i * measurements:(i + 1) * measurements]
                        shot2 = counts_shot2[i * measurements:(i + 1) * measurements]
                        atoms_loaded = [x > cutoff for x in shot1]
                        n_atoms_loaded = sum(atoms_loaded)
                        loading_fraction = n_atoms_loaded / measurements

                        if loading_fraction > 0.3:  # apparent very low rate loading might be wrongly classified background
                            cutoff = threshold_otsu(shot1)
                            atoms_loaded = [x > cutoff for x in shot1]
                            n_atoms_loaded = sum(atoms_loaded)
                            loading_fraction = n_atoms_loaded / measurements

                        n_atoms_loaded_array[i] = n_atoms_loaded
                        atoms_retained = [x > cutoff and y for x, y in zip(shot2, atoms_loaded)]
                        loading_fraction = n_atoms_loaded / measurements
                        loading_rate_array[i] = loading_fraction
                        retention_fraction = 0 if not n_atoms_loaded > 0 else sum(atoms_retained) / n_atoms_loaded

                        if retention_fraction > 0.3:
                            cutoff = threshold_otsu(shot2)
                            atoms_retained = [x > cutoff and y for x, y in zip(shot2, atoms_loaded)]
                            loading_fraction = n_atoms_loaded / measurements
                            loading_rate_array[i] = loading_fraction
                            retention_fraction = 0 if not n_atoms_loaded > 0 else sum(atoms_retained) / n_atoms_loaded

                        retention_array[i] = retention_fraction

                    error = np.array([1/np.sqrt(n) if n > 0 else 0 for n in n_atoms_loaded_array])

                    self.clear()
                    if len(x) == len(retention_array) and len(x) == len(loading_rate_array):
                        self.plot(x, retention_array,
                                  pen=None,
                                  symbol='o',
                                  symbolBrush=(255, 0, 0),
                                  symbolPen='w',
                                  name='retention')
                        self.plot(x, loading_rate_array,
                                  pen=None,
                                  symbol='o',
                                  symbolBrush=(0, 100, 100),
                                  symbolPen='w',
                                  name='loading')

                        self.setYRange(-0.0, 1.0, padding=0)

                        title = str(data[self.args.scan_vars][1])
                        self.setTitle(title)

                        if error is not None:
                            # See https://github.com/pyqtgraph/pyqtgraph/issues/211
                            if hasattr(error, "__len__") and not isinstance(error, np.ndarray):
                                error = np.array(error)
                        errbars = pyqtgraph.ErrorBarItem(
                            x=x, y=retention_array, height=2*error, pen=(255, 0, 0)) # error should be +/- the std, hence 2*
                        self.addItem(errbars)
                        self.addLegend()
            else:
                self.clear()
        except:
            self.clear()

def main():
    applet = TitleApplet(XYPlot)
    applet.add_dataset("counts_shot1", "first readout photocounts")
    applet.add_dataset("counts_shot2", "second readout photocounts")
    applet.add_dataset("measurements", "how many measurements per data point")
    applet.add_dataset("threshold_cts_per_s", "the atom signal counts/s threshold")
    applet.add_dataset("t_exposure", "the atom readout exposure time")
    applet.add_dataset("scan_vars", "the names of the scan variable(s)", required=False)
    applet.add_dataset("scan_sequence1", "the scan steps", required=False)

    applet.run()

if __name__ == "__main__":
    main()