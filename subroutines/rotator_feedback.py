"""
Optimization of polarization using motorized waveplate stages

The setup assumed is has linearly polarized light, say, |V>, which passes through a HWP, QWP,
SM fiber, a polarizer oriented to transmit |V>, and finally, a photodetector. The SM fiber is
describable as an arbitrary waveplate given by a unitary matrix (we don't care about attenuation
by the fiber).

For the plotting the datasets for 1D fits in debugging mode, use the applet command
'python "C:\\...\\qn_artiq_routines\\applets\\plot_xyline.py" FORT_PV_measured_pts
--x FORT_waveplate_angle_pts --fit FORT_PV_fit_pts --fitx FORT_fit_angle_pts
--marker FORT_estimated_max_tuple'
"""

from artiq.experiment import *
import logging
import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt

import sys, os
cwd = os.getcwd() + "\\"
sys.path.append(cwd)
sys.path.append(cwd+"\\repository\\qn_artiq_routines")

from K10CR1.KinesisMotorWrapper import KinesisMotorWrapper, KinesisMotorSimulator
from utilities.physics.polarization import V, QWP, HWP, AWP


class FORTPolarizationOptimizer:

    def __init__(self, experiment, sampler, sampler_ch, max_moves, HWP_SN='55000759', QWP_SN='55000740',
                 tolerance=0.05, debugging=False, dry_run=False):
        """
        #todo add explanatory docstring

        :param experiment: the parent experiment instance
        :param sampler: the instance of Sampler card used for measuring the FORT power
        :param sampler_ch: int representing the index of the Sampler channel
        :param max_moves: int, the maximum number of steps for an algorithm to execute before giving up
        :param HWP_SN: str, the serial number of the K10CR1 device controlling the HWP
        :param QWP_SN: str, the serial number of the K10CR1 device controlling the QWP
        :param tolerance: float, the maximum difference 1 - P_i/P_{i-1} that is required as a stopping
            condition for the feedback. I.e., if 1 - P_i/P_{i-1} <= tolerance_goal stop, where P_j is the jth
            measurement of the FORT power.
        :param debugging: bool, False by default. If true, rotator positions will be set back to their initial
            positions.
        :param dry_run: bool, whether or not to simulate the optimization routine. does not use hardware.
        """
        # the sampler ch should come from the AOM feedback channel ideally, but rather then insist
        # on reading from the json file, let's just pass in the sampler and channel when we instantiate
        # the class

        self.experiment = experiment
        self.sampler = sampler
        self.sampler_ch = sampler_ch
        self.max_moves = max_moves
        self.HWP_SN = HWP_SN
        self.QWP_SN = QWP_SN
        self.tolerance = tolerance
        self.debugging = debugging
        self.dry_run = dry_run

        if not dry_run:
            try:
                self.HWP_rotor = KinesisMotorWrapper(conn=self.HWP_SN, scale='K10CR1')
                print(f"opened K10CR1 {self.HWP_SN}")
            except Exception as e:
                logging.error(f"Failed to connect to K10CR1 {self.HWP_SN}! "
                              f"\n Check that you correctly typed the SN and that the device is connected.")
                print(self.HWP_SN, e)
                raise

            try:
                self.QWP_rotor = KinesisMotorWrapper(conn=self.QWP_SN, scale='K10CR1')
                print(f"opened K10CR1 {self.QWP_SN}")
            except Exception as e:
                logging.error(f"Failed to connect to K10CR1 {self.QWP_SN}! "
                              f"\n Check that you correctly typed the SN and that the device is connected.")
                print(self.QWP_SN, e)
                raise
        else:
            self.HWP_rotor = KinesisMotorSimulator()
            self.QWP_rotor = KinesisMotorSimulator()
            logging.info("Using simulated K10CR1 devices")

        self.HWP_initial_angle = self.HWP_rotor.get_position()
        self.QWP_initial_angle = self.QWP_rotor.get_position()

        # attributes only used in dry run
        self.awp = AWP(3.7, 1.5, 3.6) # example arb. waveplate to describe the SM fiber.
        self.n_samples1D = 10

        # initialize datasets
        self.measure_pts_dataset = "FORT_PV_measured_pts"
        self.fit_pts_dataset = "FORT_PV_fit_pts"
        self.waveplate_angle_pts_dataset = "FORT_waveplate_angle_pts"  # generic angle dataset for either waveplate
        self.fit_angle_pts_dataset = "FORT_fit_angle_pts"  # generic angle dataset for either waveplate
        self.running_maxima_dataset = "running_maxima"
        self.estimated_max_tuple_dataset = "FORT_estimated_max_tuple"  # [angle, value]
        self.experiment.set_dataset(self.measure_pts_dataset, [0.0], broadcast=True)
        self.experiment.set_dataset(self.fit_pts_dataset, [0.0], broadcast=True)
        self.experiment.set_dataset(self.waveplate_angle_pts_dataset, [0.0], broadcast=True)
        self.experiment.set_dataset(self.fit_angle_pts_dataset, [0.0], broadcast=True)
        self.experiment.set_dataset(self.running_maxima_dataset, [0.0], broadcast=True)
        self.experiment.set_dataset(self.estimated_max_tuple_dataset, [0.0, 0.0], broadcast=True)

    def final_state(self, theta, phi):
        """
        The state after the HWP, QWP, and AWP, assuming the input state is V

        :param theta: HWP angle
        :param phi: QWP angle
        :return: 2 element complex np.array giving the state in the {H,V} basis
        """
        return self.awp.dot(QWP(phi)).dot(HWP(theta)).dot(V)

    def PV(self, theta, phi) -> TFloat:
        """
        Probability of measuring V after the HWP, QWP, and AWP, assuming the input state is V
        :param theta: HWP angle
        :param phi: QWP angle
        :return: float, P(V)
        """
        return abs(np.vdot(V, self.final_state(theta, phi))) ** 2

    def get_PV_grid(self, thetas, phis):
        """
        return a simulated grid of the P(V) over arrays of theta and phi values,
        for a given arbitrary waveplate. assumes an input state of V
        """

        PV_samples = np.zeros((len(thetas), len(phis)))
        for i, theta in enumerate(thetas):
            for j, phi in enumerate(phis):
                PV_samples[i, j] = abs(np.vdot(V, self.awp.dot(QWP(phi)).dot(HWP(theta)).dot(V))) ** 2
        return PV_samples.transpose()

    @staticmethod
    def fourier_sine_series(x, *coeffs):
        """
        coeffs = [a_0, phi_0..., offset]
        so there should be 2*n+1 elements, consisting of n amplitudes, n phases, and an offset,
        where the amplitudes are the even elements (0, 2, ... 2*n-2),
        the phases are the odd elements (1, 3, ... 2*n-1), and the last element is the offset
        """
        result = np.zeros_like(x)
        for n in range(len(coeffs) // 2):
            result += coeffs[2 * n] * np.sin(n * x + coeffs[2 * n + 1])
        result += coeffs[-1]  # offset
        return result

    @rpc
    def get_estimated_maximum(self, angles: TArray(TFloat), measurements: TArray(TFloat)) -> TArray(TFloat):
        """
        Fit the measured data to a Fourier series and return max and angle for the angle that maximizes the fit

        angles: TArray(TFloat) angles in radians
        measurements: TArray(TFloat) the recorded voltages
        :return: TArray(TFloat) floats max_angle, max_PV, fit_params
        """

        p0 = np.zeros(9) # todo: could expose this in the future
        model = self.fourier_sine_series

        popt, _ = curve_fit(model, angles, measurements, p0=p0)

        # go the nearest estimated maximum P(V) value
        minimum = minimize(lambda p: -1 * model(p, *popt),
                           x0=angles[len(angles)//2], bounds=[(angles[0], angles[-1])])

        max_angle = minimum.x[0]  # update the qwp angle
        max_PV = model(minimum.x[0], *popt)

        return max_angle, max_PV, popt

    # @kernel
    def iterative_optimization(self):

        # todo
        """
        below, if not dry run:
         - theta0, phi0 -> self initial angles *pi/180
         - moves -> self.max_moves
         - do curve fitting off the kernel
         - PV -> the voltage measured from the sampler ch
         - it remains to be seen whether functions such as np.dot will work on the kernel
         - rather than go to the max position and move +/-degs, go to the nearest endpoint and do the
            entire measurement in one motion
        """

        theta0 = self.HWP_rotor.get_position()
        phi0 = self.QWP_rotor.get_position()
        theta = theta0
        phi = phi0

        phi_pts = np.zeros(self.n_samples1D)  # ARTIQ expects these to always be initialized
        theta_pts = np.zeros(self.n_samples1D)

        p0 = np.zeros(9) # fit guess

        self.theta_coords = [theta0]
        self.phi_coords = [phi0]
        running_max = [self.PV(theta0, phi0)]

        for move in range(self.max_moves):

            if not move % 2:  # move the QWP
                phi_pts = np.linspace(-np.pi / 2, np.pi / 2,
                                      self.n_samples1D) + phi  # generate pts centered on the starting angle

                self.QWP_rotor.move_to(phi_pts[0]*180/np.pi)

                # sample PV - todo: bundle this into a function?
                if self.dry_run:
                    PV_sample = np.array([self.PV(theta, p) for p in phi_pts])
                else:
                    pass  # todo measure the Sampler

                # todo: rename popt fit_params
                max_angle, estimated_PV_max, popt = self.get_estimated_maximum(phi_pts, PV_sample)

                phi = max_angle

                # running_max.append(model(phi, *popt))
                running_max.append(estimated_PV_max)

                self.phi_coords.append(phi)
                self.theta_coords.append(theta)

                if self.debugging:
                    hi_res_phi_pts = np.linspace(phi_pts[0], phi_pts[-1], 50)
                    fit_pts = self.fourier_sine_series(hi_res_phi_pts, *popt)

                    # update the datasets
                    self.experiment.set_dataset(self.measure_pts_dataset, PV_sample, broadcast=True)
                    self.experiment.set_dataset(self.waveplate_angle_pts_dataset, phi_pts, broadcast=True)
                    self.experiment.set_dataset(self.fit_pts_dataset, fit_pts, broadcast=True)
                    self.experiment.set_dataset(self.fit_angle_pts_dataset, hi_res_phi_pts, broadcast=True)
                    self.experiment.set_dataset(self.estimated_max_tuple_dataset, [phi, estimated_PV_max],
                                                broadcast=True)

            else:
                rel_pts = np.linspace(-np.pi / 2, np.pi / 2, self.n_samples1D)

                self.QWP_rotor.move_to((rel_pts[0] + phi) * 180 / np.pi)
                self.HWP_rotor.move_to((rel_pts[0] + theta) * 180 / np.pi)

                # sample PV
                if self.dry_run:
                    PV_sample = np.array([self.PV(x + theta, x + phi) for x in rel_pts])
                else:
                    pass  # todo

                max_rel_angle, estimated_PV_max, popt = self.get_estimated_maximum(rel_pts, PV_sample)

                phi += max_rel_angle
                theta += max_rel_angle

                running_max.append(estimated_PV_max)
                self.phi_coords.append(phi)
                self.theta_coords.append(theta)

                if self.debugging:
                    hi_res_angles = np.linspace(rel_pts[0], rel_pts[-1], 50)
                    fit_pts = self.fourier_sine_series(hi_res_angles, *popt)

                    # update the datasets
                    self.experiment.set_dataset(self.measure_pts_dataset, PV_sample, broadcast=True)
                    self.experiment.set_dataset(self.waveplate_angle_pts_dataset, rel_pts+phi, broadcast=True)
                    self.experiment.set_dataset(self.fit_pts_dataset, fit_pts, broadcast=True)
                    self.experiment.set_dataset(self.fit_angle_pts_dataset, hi_res_angles+phi, broadcast=True)
                    self.experiment.set_dataset(self.estimated_max_tuple_dataset, [phi, estimated_PV_max],
                                                broadcast=True)

            # P(V) is occasionally > 1 by on the order of 0.001, which is numerical error
            if abs(1 - running_max[-1]) <= self.tolerance:
                logging.info("finished in "+str(move)+" moves")
                break

        # wrap up -- if debugging, reset waveplate positions
        if self.debugging:
            self.QWP_rotor.move_to(self.QWP_initial_angle)
            self.HWP_rotor.move_to(self.HWP_initial_angle)

        if self.dry_run:
            # plot the PV map and our optimization progress
            pts = 100
            start_rad = 0
            stop_rad = 2 * np.pi
            thetas = np.linspace(start_rad, stop_rad, pts)
            phis = np.linspace(start_rad, stop_rad, pts)

            fig, ax = plt.subplots()
            PV_array = self.get_PV_grid(thetas, phis)
            cax = ax.imshow(PV_array, extent=[thetas[0], thetas[-1], phis[0], phis[-1]], origin='lower', cmap='gray')
            ax.plot(self.theta_coords, self.phi_coords, c='gold')
            ax.scatter(self.theta_coords[0], self.phi_coords[0], c='green', label='start')
            ax.scatter(self.theta_coords[-1], self.phi_coords[-1], c='red', label='stop')
            ax.legend(loc=(1.1, 0.5))
            ax.set_ylabel("QWP angles (rad.)")
            ax.set_xlabel("HWP angles (rad.)")
            # ax.set_title(
            #     r"$\theta=$" + f"{rand_theta:.2}" + r", $\phi=$" + f"{rand_phi:.2}" + r", $\eta=$" + f"{rand_eta:.2}",
            #     fontsize=8)
            ax.set_title(r'$P(V)=|\langle V|M_{\rm fiber}\cdot M_{\rm QWP}\cdot M_{\rm HWP}|V\rangle|^2$')
            plt.show()

    def run(self, method='iterative'):
        if method == 'iterative':
            self.iterative_optimization()
        elif method == 'gradient_descent':
            raise NotImplementedError
        else:
            logging.error("Polarization optimization method", method, "does not exist")
            raise


class FORTPolarizationOptimizerTest(EnvExperiment):

    def build(self):
        self.setattr_argument('dry_run', BooleanValue(True))
        self.setattr_argument('debugging', BooleanValue(True))

    def prepare(self):

        self.pol_optimizer = FORTPolarizationOptimizer(
            experiment=self, sampler=None, sampler_ch=None, max_moves=10, HWP_SN='55000759', QWP_SN='55000740',
            tolerance=0.05, debugging=self.debugging, dry_run=self.dry_run
        )

    def run(self):
        self.pol_optimizer.run()



