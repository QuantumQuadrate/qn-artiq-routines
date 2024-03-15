
from time import sleep, time
from pylablib.devices import Thorlabs # for Kinesis instrument control
import nidaqmx as daq
from artiq.experiment import *
import nidaqmx.constants as daq_constants
from nidaqmx.errors import DaqError, DaqWarning
from nidaqmx.error_codes import DAQmxErrors, DAQmxWarnings
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ArbitraryRetarder import *
from scipy.optimize import minimize
# a dictionary specifying channels we want to optimize

class RotatorFeedbackChannel():

    def __init__(self,ch_name = "Dev1/ai", dds_channel=0, rotator_sn=['55105674', '55000741'], dry_run = True ,
                 max_runs=10, leave_laser_on=False, spc = None, rate = None, plate_config = None):

        ch_name += str(dds_channel)
        self.spc = spc
        self.dry_run = dry_run
        self.daq_task = daq.Task()
        self.rate = rate

        if self.spc is None:
            self.spc = 1

        if self.rate is None or self.rate > 1e5:
            self.rate = 1e5

        self.measure_sim = plate_config_measure(plate_config)

        self.daq_task.ai_channels.add_ai_voltage_chan(physical_channel=ch_name)

        if self.dry_run:
            self.measure = self.measure_dryrun2  # use this for testing since you don't have access to artiq hardware yet
        else:
            self.measure = self._measure

        self.ser0 = rotator_sn[0]
        self.ser1 = rotator_sn[1]

        self.scl = True  # whether to use physical units - this apparently has no effect
        self.stage = [Thorlabs.KinesisMotor(conn=self.ser0, scale='K10CR1', ),
                      Thorlabs.KinesisMotor(conn=self.ser1, scale='K10CR1', )]

        self.r0 = self.stage[0]
        self.r1 = self.stage[1]
        self.r0.set_position_reference()
        self.r1.set_position_reference()

        print(f"{self._pos(0)} : {self._pos(1)}; intensity:{np.sum(self.daq_task.read(number_of_samples_per_channel=1))}")

    def print_pos(self, rotor_num=0):
        print("position = ", self._pos(rotor_num))

    def wait_stop(self, rotor_num = 2):
        if rotor_num == 2:
            self.stage[0].wait_for_stop()
            self.stage[1].wait_for_stop()
        else:
            self.stage[rotor_num].wait_for_stop()

    def is_moving(self, rotor_num = None):
        if rotor_num is None:
            return self.r0.is_moving() or self.r1.is_moving()
        else:
            return self.stage[rotor_num].is_moving()

    def print_abs_pos(self, rotor_num=0):
        print("position = ", self._abs_pos(rotor_num))

    def _pos(self, rotor_num=0):
        return self.stage[rotor_num].get_position(scale=self.scl)

    def _abs_pos(self, rotor_num=0):
        return self._pos(rotor_num) % 180

    """Using the USB-NiDaq reads out voltage measured from a Si Amplified Detector"""
    def _measure(self, measurements=None):
        if measurements is None:
            measurements = self.spc

        data = self.daq_task.read(number_of_samples_per_channel=measurements)

        return np.mean(data)


    """
    Self explanatory, generated points for two wave plates, named q and h here, represented by y0 and x0
    """

    def q_h_gen(self, range, steps, center_x, center_y, random=False):

        if not random:
            range /= 2
            rads = np.arange(0, 2 * np.pi, (2 * np.pi) / steps)
            x = range * (np.cos(rads)) - center_x
            y = range * (np.sin(rads)) - center_y
            x1 = range / 2 * (np.cos(rads)) - center_x
            y1 = range / 2 * (np.sin(rads)) - center_y
            x2 = range / 4 * (np.cos(rads)) - center_x
            y2 = range / 4 * (np.sin(rads)) - center_y
            x0 = np.append(x, x1)
            x0 = np.append(x0, x2)
            y0 = np.append(y, y1)
            y0 = np.append(y0, y2)
        if random is True:
            x0, x1, x2 = np.random.rand(steps) * range, np.random.rand(steps) * range / 2, np.random.rand(steps) * range / 4
            x0 = np.append(x0, x1)
            x0 = np.append(x0, x2)
            y0, y1, y2 = np.random.rand(steps) * range, np.random.rand(steps) * range / 2, np.random.rand(steps) * range / 4
            y0 = np.append(y0, y1)
            y0 = np.append(y0, y2)

        return x0, y0

    """
    Using pre selected point/randomly generated ones, the rotors are moved between points and then stopped.
    While the rotors are in the process of moving, we are keeping track of the angle it is at, and then the measurement.
    Due to time to measure, acceleration and speed of rotor, there is slight discrepancy between measured angle and 
    actual angle for some given intensity measurement, but the non acceleration component of the discrepancy is solved 
    for in the optimize portion of the code, and the acceleration error component seems to be negligible, when the goal
    is achieving maximum intensity.
    """
    def move_and_measure(self, theta=None, eta=None, phi=None, range=None, steps=None, a=0, b=0, theta_h=0, theta_q=0,
                         dry_run=True, E=None):
        h_ang, q_ang = self.q_h_gen(steps=steps, range=range, center_x=-a, center_y=-b, random=True)
        q_ang -= theta_q
        h_ang -= theta_h
        mq_ang = []
        mh_ang = []
        # q_ang.sort()
        # h_ang.sort()
        measure = self.measure
        measurements = []
        if dry_run or self is None:
            for h, q in zip(q_ang, h_ang):
                measurements.append(np.sum(measure(q_ang=q, h_ang=h, theta=theta, phi=phi,
                                                   eta=eta, E=E)))
        else:
            for q, h in zip(q_ang, h_ang):
                self.r1.move_to(q)
                self.r0.move_to(h)
                i = 0
                sleep(0.3)
                while self.is_moving():
                    mq_ang = np.append(mq_ang, self._pos(rotor_num=1))
                    mh_ang = np.append(mh_ang, self._pos(rotor_num=0))
                    measurement = measure(measurements=1)
                    measurements = np.append(measurements, measurement)
                    # print("intermediate measurement")
                    # print(q_ang)
                    i += 1
                    if not i % 10:
                        print("still reading...")
                    sleep(0.3)
                self.stage[1].wait_for_stop()
                self.stage[0].wait_for_stop()

        return mh_ang, mq_ang, measurements
    def move_by(self, degrees, rotor_num = -1, velocity = None, r1 = None):
        r = r1
        if rotor_num == -1 and r is None:
            r = self.stage[0]
        else:
            r = self.stage[rotor_num]
        if velocity is None:
            r.move_by(degrees)
            r.wait_move()
        else:
            print(r.get_velocity_parameters())
            r.setup_velocity(max_velocity=velocity, scale=self.scl)
            print(r.get_velocity_parameters())
            r.move_by(degrees)
            r.wait_move()
        return 0

    def max_with_tolerance(h, q, m, peak, tol):
        m_indices = [i < (peak * (1 + tol)) and i > (peak * (1 - tol)) for i in m]
        m_indices = [j == max(m) and t for j, t in zip(m, m_indices)]
        for i, j, m in zip(h, q, m):
            if i != -361 and j != -361:
                return i, j, m
        return None, None, None

    def move_to(self, degrees, rotor_num = 0, velocity = None, r1 = None):
        r = r1
        if rotor_num == -1 and r is None:
            r = self.stage[0]
        else:
            r = self.stage[rotor_num]

        r.move_to(degrees)
        r.wait_move()
        return 1


    def optimize(self, m_func, r0, r1, data=None, x0=None, bounds=None, rotor_channel=None, cons=None, range_val=180,
                 terminate=False, alpha=0, beta=0, tol=0.2):
        cons = ({'type': 'ineq',
                 'fun': constraint,
                 'args': (max(data[2]),)
                 },)

        method = 'trust-constr'
        trust_constr_opts = {'disp': True, 'barrier_tol': 1e-8, 'xtol': 1e-12,
                             'initial_constr_penalty': 1, 'maxiter': 5e3, }
        if data is not None:
            pot_max = max(data[2])
            pot_min = min(data[2])
            bounds = [(0, pi), (0, pi), (0, pi), (pot_max - pot_min, np.inf), (0, pi), (0, pi),
                      (-0.5, pot_max)]
        else:
            pot_max = rand() * 3
        if x0 is None:
            x0 = [rand() * pi, rand() * pi, rand() * pi, pot_max - pot_min, 0, 0, 0]
            sleep(1)
        result = minimize(objective_func, x0=x0, args=data, bounds=bounds, method=method,
                          constraints=cons, options=trust_constr_opts)

        x = result.x
        theta, eta, phi, E, p_q, p_h, a = x

        h = np.linspace(-range_val / 2 + alpha, range_val / 2 + alpha, 30)
        q = np.linspace(-range_val / 2 + beta, range_val / 2 + beta, 30)

        X, Y = np.meshgrid(h, q)

        Z1 = measure(q_ang=Y, h_ang=X, theta=theta, eta=eta, phi=phi, theta_q=p_q, theta_h=p_h, E=E, a=a)
        peak = (E + a)

        c = np.argmax(Z1)

        maxX, maxY, maxZ = X[c // 30][c % 30], Y[c // 30][c % 30], Z1[c // 30][c % 30]

        r0.move_to(maxX)
        r1.move_to(maxY)
        r0.wait_for_stop()
        r1.wait_for_stop()
        test_measurement = m_func(measurements=1000)
        sleep(1)
        print((abs(test_measurement - peak) / peak))
        within_range = test_measurement >= peak * (1 - tol) and test_measurement <= peak * (1 + tol)
        terminate = within_range or terminate
        if not terminate:
            newh, newq, newm = self.move_and_measure(range=90, steps=5, r_feedback=rotor_channel, dry_run=False, a=maxX,
                                                b=maxY)
            h, q, m = data
            h = np.append(h, newh)
            q = np.append(q, newq)
            m = np.append(m, newm)
            data = h, q, m

            checkMaxH, checkMaxQ, checkMaxM = self.max_with_tolerance(*data, peak, tol)
            if checkMaxH is not None and checkMaxQ is not None:
                r0.move_to(maxX)
                r1.move_to(maxY)
                r0.wait_for_stop()
                r1.wait_for_stop()
                test_measurement = m_func(measurements=1000)
                sleep(1)
                print((abs(test_measurement - peak) / peak))
                within_range = test_measurement >= peak * (1 - tol) and test_measurement <= peak * (1 + tol)
                if (within_range):
                    return X, Y, Z1, checkMaxH, checkMaxQ, checkMaxM
            return self.optimize(m_func=m_func, r0=r0, r1=r1, x0=None, data=data, rotor_channel=rotor_channel,
                            terminate=True, tol=tol)
        else:
            return X, Y, Z1, maxX, maxY, maxZ
    def close(self):
        self.stage[0].close()
        self.stage[1].close()
        self.daq_task.close()
        print("Closed successfully")

    def timed_scan_test(self, stage, speed = 10, range = 90, start_pos = 0, max = None):
        acceleration = speed
        range_array = self.get_range(stage.get_position(), range/2)
        stage.move_to(range_array[0] - 1 / 2 * acceleration)
        stage.wait_for_stop()
        print(stage.get_position())
        stage.setup_velocity(max_velocity=speed, acceleration=acceleration)
        stage.move_to(start_pos+range/2)
        degree_range = []

        degree_range.append(stage.get_position())
        intensity_range = [-1]
        while stage.is_moving() and stage.get_position() <= range_array[1]:
            pos = stage.get_position()
            degree_range.append(pos)
            x = self.measure()
            intensity_range.append(x)
            if x >= np.max(intensity_range):
                deg_max = pos
        print(deg_max)
        stage.move_to(deg_max)
        stage.wait_for_stop()
        return self.slow_read(stage = stage, degree_stage=degree_range,
                              intensity_stage=intensity_range)

    def in_range(self, stage, range):
        pos = stage.get_position()
        return pos >= range[0] and pos <= range[1]



    def quick_move_read_test(self, rotor_num, init_pos = None, speed = 10, range = 45, full_optimize = False):
        if full_optimize:
            range = 45
        acceleration = speed
        interval = 0.01
        stage = self.stage[rotor_num]
        stage.stop()
        stage.wait_for_stop()
        degree_stage = [stage.get_position()]
        intensity_stage = [self.measure()]
        range_array = self.get_range(degree_stage[0], pmrange=range)
        print(range_array)
        sleep(2)
        init_pos = range_array[0] - 1/2*acceleration
        stage.setup_velocity(max_velocity=10, acceleration=10)
        stage.move_to(init_pos)
        stage.wait_for_stop()
        stage.setup_velocity(max_velocity=speed, acceleration=acceleration)

        stage.jog(direction="+")
        moving_forward = True
        moving = True
        sleep(1)

        i=0
        try:
            while moving:

                degree_before = stage.get_position()
                time_before_measure = time()
                degree = stage.get_position()
                intensity = self.measure()
                degree_dif = (time_before_measure-time())*speed
                in_range = self.in_range(stage, range_array)
                print(f"in_range: Quick_read :{in_range}")
                if degree_before >= degree:
                    degree -= degree_dif
                else:
                    degree += degree_dif

                if not in_range and i == 0:
                    stage.stop()
                    stage.wait_for_stop()
                    stage.setup_velocity(max_velocity=speed, acceleration=acceleration)
                    if moving_forward is True:
                        stage.jog(direction="-")
                        moving_forward =  False
                    else:
                        stage.jog(direction="+")

                    while (self.in_range(stage, range_array) is not True):
                        sleep(0.01)
                i+=1
                print(f"Iteration {i}\n Degree: {degree}"
                      f"\nIntensity: {intensity}"
                      f"\n Time to measure: "
                      f""
                      f"\n^^^^^^^^^^^^^^^^^^^^^^^^^")


                if not in_range and moving_forward is not True:
                    stage.stop()
                    stage.wait_for_stop()
                    moving = False
                if not in_range and moving_forward is True:
                    stage.stop()
                    stage.wait_for_stop()
                    sleep(1)
                    moving_forward = False
                    i=0
                degree_stage.append(degree)
                intensity_stage.append(intensity)
                sleep(interval)
        except Exception as e:
            print(e)
            stage.stop()
            stage.wait_for_stop()

        max_val = max(intensity_stage)
        degree_max = degree_stage[intensity_stage.index(max_val)]
        stage.move_to(degree_max)

        return degree_stage, intensity_stage
    def get_range(self,position,pmrange):
        return [position-pmrange, position+pmrange]
    def slow_read(self, stage, degree_stage = [], scan_range = 2, intensity_stage = [],speed = 0.2, is_optimized = False,):
        stage.wait_for_stop()
        true_diff = -100000



        acceleration = speed
        interval = 0.1
        intensity = self.measure

        max_degree = stage.get_position()
        init_pos = max_degree
        max_intensity = self.measure()
        degree_stage.append(max_degree)
        intensity_stage.append(max_intensity)

        if scan_range >= 90:
            interval/=2
            scan_range = 90
        range = self.get_range(init_pos,scan_range/2)


        if speed > 10:
            speed = 10
            acceleration = 10



        stage.setup_velocity(max_velocity=10, acceleration=10)
        stage.move_to(range[0] - acceleration/2)
        stage.wait_for_stop()

        stage.setup_velocity(max_velocity=speed, acceleration=acceleration)

        stage.jog(direction="+")
        sleep(1)

        degree = stage.get_position
        moving = True
        i = 0
        try:
            while moving:
                print(1)
                degree_before = degree()
                time_before_measure = time()
                degree_measured = degree()
                i_measured = intensity()
                if degree_before >= degree_measured:
                    degree_measured -= (time_before_measure-time())*speed
                else:
                    degree_measured += (time_before_measure-time())*speed

                in_range = degree_measured >= range[0] and degree_measured <= range[1]
                if i == 0 and not in_range:
                    stage.stop()
                    stage.wait_for_stop()

                    stage.setup_velocity(max_velocity=10, acceleration=10)
                    stage.move_to(range[0]-acceleration/2)
                    stage.wait_for_stop()

                    stage.setup_velocity(max_velocity=speed, acceleration=acceleration)
                    print(f"degree: {degree_measured},range {range}\nin range:{in_range}")
                    stage.jog(direction="-")
                    sleep(1)
                elif in_range:
                    print(f"degree: {degree_measured}, intensity{i_measured}")
                    degree_stage.append(degree_measured)
                    intensity_stage.append(i_measured)
                    i+=1
                else:
                    print(f"degree: {degree_measured}, intensity{i_measured}, range {range}")
                    degree_stage.append(degree_measured)
                    intensity_stage.append(i_measured)
                    stage.stop()
                    moving = False
                    stage.wait_for_stop()
                sleep(interval)
        except Exception as e:
            print(f"{e}::error")
            stage.stop()
            stage.wait_for_stop()

        max_degree_found, max_intensity = degree_stage[intensity_stage.index(max(intensity_stage))], max(intensity_stage)

        stage.setup_velocity(max_velocity=10, acceleration=10)
        stage.move_to(max_degree_found)
        stage.wait_for_stop()

        start_max_diff = float(abs(max_degree_found-init_pos))
        if len(intensity_stage) >=2:
            max_val, second_max = sorted(intensity_stage)[-2:]
            degree_max = degree_stage[intensity_stage.index(max_val)]
            degree_second_max = degree_stage[intensity_stage.index(second_max)]
            true_diff = degree_max - degree_second_max



        if ((start_max_diff)/90 >= 0.0005) and true_diff/90 >=0.0005:
            return self.slow_read(stage = stage, degree_stage = degree_stage, intensity_stage = intensity_stage,
                                  scan_range = scan_range*5, speed=speed*5, is_optimized = False)

        elif ((start_max_diff)/90 >= 0.0005):
            return self.slow_read(stage=stage, degree_stage=degree_stage, intensity_stage=intensity_stage,
                                    scan_range=scan_range / 3, speed=speed / 3,
                                    is_optimized=True)
        else:
            return max_degree_found, max_intensity, degree_stage, intensity_stage


    def new_optimize_test(self, rotor_num, init_pos = None, quick_speed = 10, slow_speed = 0.5, full_optimize = False,
                          is_optimized = False):
        max_degree = 0
        intensity_max = 0
        if is_optimized is False:
            degree_stage, intensity_stage = self.quick_move_read_test(rotor_num=rotor_num, speed=quick_speed,
                                                                      full_optimize = full_optimize)

            max_degree, intensity_max, degree_stage, intensity_stage = self.slow_read(stage=self.stage[rotor_num], degree_stage = degree_stage
                                                       ,intensity_stage = intensity_stage, speed = slow_speed,)
            self.stage[rotor_num].setup_velocity(max_velocity=10, acceleration = 10)
            self.stage[rotor_num].move_to((max_degree)%360)
            self.stage[rotor_num].wait_for_stop()
        else:
            max_degree, intensity_max, degree_stage, intensity_stage = self.slow_read(stage=self.stage[rotor_num], scan_range = 1,
                                               speed=slow_speed, is_optimized = is_optimized)
        return max_degree, intensity_max, degree_stage, intensity_stage

def test():
    rotator_feedback_dict = {
        'ch_1':
            {
                "ch_name": "dipole_trap1",
                "dds_channel": "1",  # this is basically the laser we want to turn on. don't worry about this
                "rotator_sn": 0o012345,  # the rotator id or serial number
                "sampler_ch": "1"  # the channel which reads in the voltage from a detector. don't worry about this
            },
        'ch_2':
            {
                "ch_name": "dipole_trap2",
                "dds_channel": "2",
                "rotator_sn": 0o023456,  # the rotator id or serial number
                "sampler_ch": "2"  # the channel which reads in the voltage from a detector. don't worry about this
            }
    }
    def intensity_sim(degree, phase, A = 1):
        intensity = A*np.sin((np.pi*np.float64(degree-phase))/(180))
        return intensity

    """"""
    stage = Thorlabs.KinesisMotor(conn="55105674", scale='K10CR1')
    #stage.move_by(100)
    #stage.wait_move()
    stage.stop()
    stage.wait_for_stop()
    sleep(1)
    phase = 45

    stage.move_to(position=132)
    stage.wait_for_stop()
    stage.jog(direction = "+")

    i = 0

    degree = stage.get_position()
    print(degree)
    i_max = intensity_sim(degree=degree, phase=phase)
    max_degree = -361
    min_possible = -361
    max_possible = 180
    reversals = 0
    moving_forward = True
    factor = 2
    time_wait = 0.1
    while stage.is_moving():
        degree = stage.get_position()
        intensity = intensity_sim(degree=degree, phase=phase)

        if intensity >= i_max:
            i_max = intensity
            max_degree = degree
            was_above = True

        elif intensity <= i_max/factor or (degree >= max_possible or degree <= min_possible):
            factor/=1.1
            if factor <= 1:
                factor = 1.001

            stage.stop()

            if moving_forward is True:

                stage.jog(direction="-")
                moving_forward = False
                #max_possible = degree
                sleep(1)
                """while d > degree - (45.0 / 2.0 ** reversals):
                    d = stage.get_position()
                    sleep(0.1)"""
            else:
                stage.stop()
                stage.wait_for_stop()
                stage.jog(direction="+")
                moving_forward = True
                #min_possible = degree
                sleep(1)
                """while d < (degree + 45.0 / 2.0 ** reversals):
                    d = stage.get_position()
                    sleep(0.1)"""

            reversals += 1
            time_wait = (0.2/(reversals))
            if reversals == 10:

                stage.move_to(position=max_degree)
                stage.wait_for_stop()

        sleep(time_wait)
        i+=1
        if i == 10000:
            stage.stop()

    print(f"Max degree:{max_degree}\nIntensity at degree:{i_max}")
    stage.close()

      # don't do anything until the motion is complete
    #rotor1.print_pos()
    #rotor1.test()

class RotorExperiment(EnvExperiment):

    def run(self):
        #devices = Thorlabs.list_kinesis_devices()  # can use this to print out the detected devices
        #print(devices)
        rotor1 = RotatorFeedbackChannel(ch_name="Dev1/ai0", rotator_sn=["55105674", "55000741"], dry_run=False)
        rotor1.stage[0].setup_velocity(max_velocity=10)
        rotor1.stage[1].setup_velocity(max_velocity=10)
        rotor1.stage[0].stop()
        rotor1.stage[1].stop()
        #rotor1.stage[0].move_to(0)
        #rotor1.stage[1].move_to(0)


        #print(rotor1.new_optimize_test(0, is_optimized=True, slow_speed=0.25,full_optimize=True))
        #print(rotor1.new_optimize_test(1, is_optimized=True, slow_speed=0.25, full_optimize=True))
        #i_max1, degree1 = rotor1.slow_optimize_rotor(1)
        #rotor1.stage[1].wait_for_stop()
        #i_max0, degree0 = rotor1.slow_optimize_rotor(0)
        #print(rotor1.timed_scan_test(rotor1.stage[0],))
        max_degree, max_intensity,degree_stage, intensity_stage = rotor1.timed_scan_test(stage=rotor1.stage[0],speed=5 )
        plt.scatter(degree_stage[1:], intensity_stage[1:])
        plt.show()

        #max_degree, max_intensity, degree_stage, intensity_stage = \
        #    rotor1.new_optimize_test(rotor_num=0,is_optimized = False, full_optimize=True)

        #plt.scatter(degree_stage[1:], intensity_stage[1:])
        #plt.show()
        #max_degree1, max_intensity1, degree_stage1, intensity_stage1 = \
        #    rotor1.new_optimize_test(rotor_num=1, is_optimized=False, full_optimize=True)

        #plt.scatter(degree_stage1[1:], intensity_stage1[1:])
        #plt.show()

        max_degree, max_intensity, degree_stage, intensity_stage = \
            rotor1.new_optimize_test(rotor_num=0, is_optimized=True, full_optimize=True)
        plt.scatter(degree_stage[1:], intensity_stage[1:])
        plt.show()
        max_degree1, max_intensity1, degree_stage1, intensity_stage1 = \
            rotor1.new_optimize_test(rotor_num=1, is_optimized=True, full_optimize=True)

        plt.scatter(degree_stage1[1:], intensity_stage1[1:])
        plt.show()

        rotor1.stage[0].wait_for_stop()
        rotor1.stage[1].wait_for_stop()
        #rotor1.measure_dry_run2()
        #rint(f"For rotor 0: {i_max0}, {degree0}")
        #rint(f"For rotor 1: {i_max1}, {degree1}")
        rotor1.close()






