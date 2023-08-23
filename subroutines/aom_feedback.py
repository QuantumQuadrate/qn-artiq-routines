"""
# todo: this should eventually replace stabilizer.py

A subroutine feedback loop using Urukul channels and Samplers.
While ARTIQ has a Servo instrument for doing feedback, it does not allow
using the Urukul and Sampler channels for other purposes. I.e. it is for
continuous noise-eating. This code is for adjusting DDS power only when
we want to, and doesn't prevent us from using some channels on a given
Sampler for other purposes.

Intended usage:
1. Instantiate the servo in the prepare method of the parent artiq experiment
2. Call run() during the parent experiment loop whenever it is desired to tune up
the AOM powers.

See the example experiment toward the bottom of this file.

Preston's notes:
todo: should be able to modify setpoints so we can optimize them as needed.
one way to do this would be to make the stabilizer dict a json file and
update the json file, e.g. after doing an optimizer routine with the AOM
powers. then whatever the voltages are after doing this routine, we update
the file. however, it might be preferable to scan over the setpoints
themselves (with proper caps on the AOM RF power), in which case, they
should be experiment variables. not obvious how to do this. the MOT can
likely be moved only using the shim fields, but the beam pairs themselves
will need to be precisely balanced in order to do good PGC. so adjusting
these setpoints will be a necessity in a matter of time.
"""

from artiq.experiment import *
import numpy as np
import time

# group all dds channels and feedback params by sampler card
# todo: set_point should reference an ExperimentVariable name
#  and the set_point should be updated at the beginning of the run method
stabilizer_dict = {
    'sampler0':
        {
            'dds_cooling_DP': # signal monitored by PD0
                {
                    'sampler_ch': 0, # the channel connected to the appropriate PD
                    # 'transfer_function': lambda x : x, # converts volts to optical mW
                    'set_point': 0.744, # volts wrt to background
                    'p': 0.07, # the proportionality constant
                    'series': False, # if series = True then these channels are fed-back to one at a time
                    'dataset':'MOT_switchyard_monitor'
                },
            'dds_AOM_A5': # signal monitored by PD5
                {
                    'sampler_ch': 1, # the channel connected to the appropriate PD
                    # 'transfer_function': lambda x : x,
                    'set_point': 2.0, # volts wrt to background
                    'p': 0.07, # the proportionality constant
                    'series': True,
                    'dataset':'MOT5_monitor'
                },
            'dds_AOM_A6': # signal monitored by PD6
                {
                    'sampler_ch': 2, # the channel connected to the appropriate PD
                    # 'transfer_function': lambda x : x,  # arbitrary units
                    'set_point': 0.288, # volts wrt to background
                    'p': 0.07, # the proportionality constant
                    'series': True,
                    'dataset': 'MOT6_monitor'
                },
            'dds_AOM_A1': # signal monitored by Femto fW detector
                {
                    'sampler_ch': 3, # the channel connected to the appropriate PD
                    # 'transfer_function': lambda x : x,
                    'set_point': 0.922, # volts wrt to background
                    'p': 0.07, # the proportionality constant
                    'series': True,
                    'dataset': 'MOT1_monitor'
                },
            'dds_AOM_A2': # signal monitored by Femto fW detector
                {
                    'sampler_ch': 3, # the channel connected to the appropriate PD
                    # 'transfer_function': lambda x : x,  # arbitrary units
                    'set_point': 0.904, # volts wrt to background
                    'p': 0.07, # the proportionality constant
                    'series': True,
                    'dataset': 'MOT2_monitor'
                },
            'dds_AOM_A3': # signal monitored by Femto fW detector
                {
                    'sampler_ch': 3, # the channel connected to the appropriate PD
                    # 'transfer_function': lambda x : x,
                    'set_point': 1.436, # volts wrt to background
                    'p': 0.07, # the proportionality constant
                    'series': True,
                    'dataset': 'MOT3_monitor'
                },
            'dds_AOM_A4': # signal monitored by Femto fW detector
                {
                    'sampler_ch': 3, # the channel connected to the appropriate PD
                    # 'transfer_function': lambda x : x,
                    'set_point': 0.844, # volts wrt to background
                    'p': 0.07, # the proportionality constant
                    'series': True,
                    'dataset': 'MOT4_monitor'
                }
        },
    'sampler1':
        {
        }
}

class FeedbackChannel:

    def __init__(self, name, dds_obj, buffer_index, set_point, p, dataset):
        """
        class which defines a DDS feedback channel

        parameters:
        'name': the name of the dds channel, e.g. dds_AOM_A1
        'dds_obj': the experiment reference to the dds object with name 'name', e.g. experiment.dds_AOM_A1
        'g(x: float)->float': a callable that converts the measured voltage to the units of the setpoint, e.g. mW
        'set_point': the float value we are trying to reach
        'p': the proportional constant for feedback
        """
        self.name = name
        self.dds_obj = dds_obj
        self.buffer_index = buffer_index
        # self.g = g
        self.set_point = set_point
        self.p = p
        self.frequency = 100*MHz
        self.amplitude = 0.0
        self.value = 0.0 # the last value of the measurement
        self.value_normalized = 0.0 # self.value normalized to the set point
        self.dataset = dataset

    @rpc(flags={"async"})
    def set_value(self, value):
        self.value = value
        self.value_normalized = value / self.set_point
        print(value)

    @rpc(flags={"async"})
    def print(self, x):
        print(x)

    @kernel
    def get_dds_settings(self):
        """ get the frequency and amplitude """
        self.frequency, _, self.amplitude = self.dds_obj.get()

    @kernel
    def feedback(self, buffer):
        """ feedback to this channel
        buffer: a list of length storing the measurement result from all samplers
        """

        # measured = self.g(buffer[self.buffer_index]) # the value of the sampler ch in setpoint units
        measured = buffer[self.buffer_index]
        err = self.set_point - measured
        ampl = self.amplitude + self.p*err

        self.value = measured
        self.value_normalized = measured/self.set_point
        # self.set_value(measured)

        # todo print some warning to alert the user if we couldn't reach the setpoint,
        if ampl < 0:
            self.amplitude = 0.0
        elif ampl > 0.9:
            self.amplitude = 0.9
        else:  # valid amplitude for DDS
            self.amplitude = ampl

        # update the dds amplitude
        # self.dds_obj.set(frequency=self.frequency,amplitude=self.amplitude)


class AOMPowerStabilizer2:

    def __init__(self, experiment, dds_names, iterations=10, t_meas_delay=10 * ms):
        """
        An experiment subsequence for reading a Sampler and adjusting Urukul output power.

        parameters:
        'experiment': a class inheriting from artiq.experiment.EnvExperiment. It is
        assumed that any hardware objects that are referenced in this class are bound to
        this instance of experiment.
        'dds_names': a list of the names of the dds channels to feedback to
        'iterations': integer number of feedback cycles to converge to the setpoints
        't_meas_delay': time to wait between iterations. mostly for avoiding underflow errors.
        """

        # initialized by user
        self.exp = experiment
        self.iterations = iterations # number of times to adjust dds power per run() call
        self.dds_names = dds_names # the dds channels for the AOMs to stabilize
        self.t_meas_delay = t_meas_delay
        self.measurement_buffer = [] # a buffer to store the measurement from all sampled Samplers
        self.sampler_list = [] # stores the list of Sampler references
        self.series_channels = [] # the feedback channels that should be adjusted one-by-one
        self.parallel_channels = []
        self.last_measurement = {}

        # this block instantiates FeedbackChannel objects and groups them by series/parallel feedback flag
        for i, sampler_name in enumerate(stabilizer_dict):

            # the dds channel names associated with this sampler
            feedback_channels = stabilizer_dict[sampler_name]

            # check if any of the dds names is associated with this sampler
            if True in [dds_name in feedback_channels.keys() for dds_name in dds_names]:

                # self.measurement_buffer += [0.0] * 8
                self.sampler_list.append(getattr(self.exp, sampler_name))

                # loop over the dds channels associated with this sampler
                for ch_name in feedback_channels.keys():

                    # create a feedback object for dds channels we want to feed back to
                    if ch_name in dds_names:

                        ch_params = feedback_channels[ch_name]
                        print(f"adding FeedbackChannel {ch_name} with dataset "
                              f"{ch_params['dataset']}")

                        fb_channel = FeedbackChannel(name=ch_name,
                                            dds_obj=getattr(self.exp, ch_name),
                                            buffer_index=ch_params['sampler_ch'] + i*8,
                                            # g=ch_params['transfer_function'],
                                            set_point=ch_params['set_point'],
                                            p=ch_params['p'],
                                            dataset=ch_params['dataset']
                        )

                        if ch_params['series']:
                            self.series_channels.append(fb_channel)
                        else:
                            self.parallel_channels.append(fb_channel)

        # self.measurement_buffer = np.array(self.measurement_buffer)
        # self.background_buffer = self.measurement_buffer
        self.measurement_buffer = np.array([0.0]*8*len(self.sampler_list))
        self.background_buffer = np.array([0.0]*8*len(self.sampler_list))

        self.all_channels = self.parallel_channels + self.series_channels

        # for logging the measured voltages
        for ch in self.all_channels:
            self.exp.set_dataset(ch.dataset, [1.0], broadcast=True)

        # the number of pts to plot
        self.exp.set_dataset('MOT_rolling_pts', [10], broadcast=True)


    @rpc(flags={"async"})
    def print(self, x):
        print(x)

    @kernel
    def measure(self):
        """
        measure all Sampler cards used for feedback
        # todo: could add option to average for better snr
        """
        i = 0
        for sampler in self.sampler_list:
            sampler.sample(self.measurement_buffer[i:i + 8])
            delay(1*ms)
            i+=1

    @kernel
    def measure_background(self):
        """
        measure all Sampler cards used for feedback
        # todo: could add option to average for better snr
        """
        i = 0
        for sampler in self.sampler_list:
            sampler.sample(self.background_buffer[i:i + 8])
            delay(1*ms)
            i+=1


    # dry run test
    # def run(self):
    #     # update the datasets
    #     for i, ch in enumerate(self.all_channels):
    #         x = np.random.rand()/10 + i
    #         print(f"updating {ch.dataset} with {x}")
    #         self.exp.append_to_dataset(ch.dataset, x)
    #         time.sleep(0.01)
    #     time.sleep(0.5)

    @kernel
    def run(self):
        """
        Run the feedback loop. On exiting, this function will turn off all dds channels
         given by dds_names. If any beams which need to be adjusted in series are in
         dds_names, all such channels will be turned off, i.e. even ones we are not
         feeding back to.
        """

        # self.exp.core.break_realtime()

        for ch in self.all_channels:
            ch.get_dds_settings()
            delay(1*ms)
            ch.dds_obj.sw.off()
            delay(1*ms)

        self.print("turned off all channels")

        with sequential:

            self.measure_background() # this updates the background list
            for ch in self.parallel_channels:
                ch.dds_obj.sw.on()
            delay(1*ms)

            # self.print("background:")
            # self.print(self.background_buffer)

            # do feedback on the "parallel" channels
            for i in range(self.iterations):
                self.measure()
                delay(10 * ms)
                # with parallel: # strictly speaking, doesn't really matter if these are parallel
                for ch in self.parallel_channels:
                    self.print(ch.name)
                    self.print(self.background_buffer)
                    self.print(self.measurement_buffer)
                    ch.feedback(self.measurement_buffer - self.background_buffer)
                delay(1 * ms)

            for ch in self.parallel_channels:
                ch.dds_obj.sw.off()
            delay(1*ms)

            self.print("fed back parallel channels")

            self.measure_background()

            # do feedback on the series channels
            with sequential:
                for ch in self.series_channels:
                    ch.dds_obj.sw.on()
                    delay(10 * ms)
                    for i in range(self.iterations):
                        self.measure()
                        delay(1*ms)
                        ch.feedback(self.measurement_buffer - self.background_buffer)
                        delay(1*ms)
                    ch.dds_obj.sw.off()

            self.print("fed back series channels")

            # update the datasets
            for ch in self.all_channels:
                self.print(ch.dataset)
                self.print(ch.value_normalized)
                self.exp.append_to_dataset(ch.dataset, ch.value_normalized)
            self.print("updated all datasets")