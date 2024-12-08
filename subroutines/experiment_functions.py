from artiq.experiment import *
import logging
import numpy as np
import pyvisa as visa

import os, sys
cwd = os.getcwd() + "\\"

sys.path.append(cwd)
sys.path.append(cwd+"\\repository\\qn_artiq_routines")

from utilities.conversions import dB_to_V_kernel as dB_to_V

"""
Table of contents:
1. Subroutine functions
2. Experiment functions
3. Diagnostic functions
"""

###############################################################################
# 1. SUBROUTINE FUNCTIONS
# These are functions that get used within various experiments. they are not
# intended to be run in a standalone fashion, e.g. from GeneralVariableScan.
# Consequently, note that the name should not end in "experiment"
###############################################################################

@kernel
def load_MOT_and_FORT(self):
    """
    The FORT loading sequence, from loading a MOT to hopefully trapping one atom

    Loads the MOT then the turns on the FORT and waits a fixed amount of time
    to load an atom. Optionally, the FORT can be turned on right away if
    FORT_on_at_start is True.

    :param self: the experiment instance
    :return:
    """

    self.dds_FORT.sw.on()

    if not self.FORT_on_at_MOT_start:
        self.dds_FORT.sw.off()
    else:
        self.dds_FORT.set(frequency=self.f_FORT, amplitude=self.stabilizer_FORT.amplitude)

    # set the cooling DP AOM to the MOT settings
    self.dds_cooling_DP.set(frequency=self.f_cooling_DP_MOT, amplitude=self.ampl_cooling_DP_MOT)

    # self.ttl7.pulse(self.t_exp_trigger)  # in case we want to look at signals on an oscilloscope

    self.dds_cooling_DP.sw.on()

    delay(1*ms)

    self.dds_AOM_A1.sw.on()
    self.dds_AOM_A2.sw.on()
    self.dds_AOM_A3.sw.on()
    self.dds_AOM_A4.sw.on()
    self.dds_AOM_A5.sw.on()
    self.dds_AOM_A6.sw.on()

    delay(1*ms) # if this delay is not here, the following line setting the dac doesn't execute

    # todo: delete
    # self.print_async("in load MOT:",self.AZ_bottom_volts_MOT, self.AZ_top_volts_MOT, self.AX_volts_MOT, self.AY_volts_MOT)

    # Turn on the MOT coils and cooling light
    self.zotino0.set_dac(
        [self.AZ_bottom_volts_MOT, self.AZ_top_volts_MOT, self.AX_volts_MOT, self.AY_volts_MOT],
        channels=self.coil_channels)

    self.ttl_UV.pulse(self.t_UV_pulse)

    # wait for the MOT to load
    delay(self.t_MOT_loading - self.t_MOT_phase2)

    if self.t_MOT_phase2 > 0:

        self.zotino0.set_dac(
            [self.AZ_bottom_volts_MOT_phase2, self.AZ_top_volts_MOT_phase2, self.AX_volts_MOT_phase2,
             self.AY_volts_MOT_phase2],
            channels=self.coil_channels)

        self.dds_cooling_DP.set(frequency=self.f_cooling_DP_MOT_phase2,
                                amplitude=self.ampl_cooling_DP_MOT) # todo: make a variable for phase 2
        delay(self.t_MOT_phase2)

    # turn on the dipole trap and wait to load atoms
    self.dds_FORT.set(frequency=self.f_FORT, amplitude=self.stabilizer_FORT.amplitude)

    if not self.FORT_on_at_MOT_start:
        delay_mu(self.t_FORT_loading_mu)

    self.stabilizer_FORT.run(setpoint_index=1) # the science setpoint

    self.dds_cooling_DP.sw.off()
    delay(self.t_MOT_dissipation)  # should wait several ms for the MOT to dissipate
    self.ttl_SPCM_gate.off()
    t_gate_end = self.ttl_SPCM0.gate_rising(self.t_SPCM_first_shot)
    self.counts_FORT_science = self.ttl_SPCM0.count(t_gate_end)
    delay(1*ms)


    if self.do_PGC_in_MOT and self.t_PGC_in_MOT > 0:
        self.dds_cooling_DP.sw.on()
        self.dds_FORT.set(frequency=self.f_FORT,
                          amplitude=self.stabilizer_FORT.amplitudes[1])

        self.zotino0.set_dac([self.AZ_bottom_volts_PGC, self.AZ_top_volts_PGC, self.AX_volts_PGC, self.AY_volts_PGC],
                             channels=self.coil_channels)

        self.dds_cooling_DP.set(frequency=self.f_cooling_DP_PGC,
                                amplitude=self.ampl_cooling_DP_MOT*self.p_cooling_DP_PGC)
        delay(self.t_PGC_in_MOT)
        self.dds_cooling_DP.sw.off()


@kernel
def load_MOT_and_FORT_for_Luca_scattering_measurement(self):
    """
    Modified version of load_MOT_and_FORT for imaging scattering in the chamber with the Luca
    and with an APD. Assumes the APD is connected to the Sampler card and channel specified in
    the code below.

    I typically run this with the MonitorFORTWithLuca experiment.

    The following shots are taken with the Luca:
    1. FORT scattering after feedback to loading setpoint
    2. FORT + MOT beam scattering

    :param self: the experiment instance
    :return:
    """

    self.dds_FORT.sw.on()

    if self.MOT_repump_off or self.MOT_light_off: # this is useful so that the photocounts dataset is all background
        self.ttl_repump_switch.on()  # turns the RP AOM off
    if self.MOT_light_off:
        self.dds_cooling_DP.sw.off()

    if not self.FORT_on_at_MOT_start:
        self.dds_FORT.sw.off()
    else:
        self.dds_FORT.set(frequency=self.f_FORT, amplitude=self.stabilizer_FORT.amplitude)

    # record FORT scattering with Luca and record Raman scattering from SM fiber
    self.ttl_Luca_trigger.pulse(5 * ms) # FORT loading scattering shot
    self.ttl_SPCM_gate.off()
    t_gate_end = self.ttl_SPCM0.gate_rising(self.t_SPCM_first_shot)
    self.counts_FORT_loading = self.ttl_SPCM0.count(t_gate_end)
    delay(10*us)

    self.APD_FORT_volts_loading = 0.0
    for i in range(self.APD_averages):
        self.sampler1.sample(self.APD_buffer)
        self.APD_FORT_volts_loading += self.APD_buffer[7]
        delay(1*ms)
    self.APD_FORT_volts_loading /= self.APD_averages

    delay(1 * ms)

    # set the cooling DP AOM to the MOT settings
    self.dds_cooling_DP.set(frequency=self.f_cooling_DP_MOT, amplitude=self.ampl_cooling_DP_MOT)

    # self.ttl7.pulse(self.t_exp_trigger)  # in case we want to look at signals on an oscilloscope

    if not self.MOT_light_off:
        self.dds_cooling_DP.sw.on()

        delay(1*ms)

        self.dds_AOM_A1.sw.on()
        self.dds_AOM_A2.sw.on()
        self.dds_AOM_A3.sw.on()
        self.dds_AOM_A4.sw.on()
        self.dds_AOM_A5.sw.on()
        self.dds_AOM_A6.sw.on()

    delay(1*ms) # if this delay is not here, the following line setting the dac doesn't execute

    # Turn on the MOT coils and cooling light
    self.zotino0.set_dac(
        [self.AZ_bottom_volts_MOT, self.AZ_top_volts_MOT, self.AX_volts_MOT, self.AY_volts_MOT],
        channels=self.coil_channels)

    # wait for the MOT to load
    delay(self.t_MOT_loading/2)
    self.ttl_Luca_trigger.pulse(5 * ms) # total scattering shot
    self.ttl_SPCM_gate.off()
    t_gate_end = self.ttl_SPCM0.gate_rising(self.t_SPCM_first_shot)
    self.counts_FORT_and_MOT = self.ttl_SPCM0.count(t_gate_end)
    delay(1 * ms)
    delay(self.t_MOT_loading/2)

    # turn on the dipole trap and wait to load atoms if it is not already on
    self.dds_FORT.set(frequency=self.f_FORT, amplitude=self.stabilizer_FORT.amplitude)

    if not self.FORT_on_at_MOT_start:
        delay_mu(self.t_FORT_loading_mu)
    delay(1*ms)
    if not self.no_feedback:
        self.stabilizer_FORT.run(setpoint_index=1) # set to  science setpoint

    self.dds_cooling_DP.sw.off()
    delay(1*ms)
    # record FORT scattering with Luca and record Raman scattering from SM fiber
    self.ttl_Luca_trigger.pulse(5 * ms)  # FORT loading scattering shot
    self.ttl_SPCM_gate.off()
    t_gate_end = self.ttl_SPCM0.gate_rising(self.t_SPCM_first_shot)
    self.counts_FORT_science = self.ttl_SPCM0.count(t_gate_end)
    delay(10*us)

    self.APD_FORT_volts_science = 0.0
    for i in range(self.APD_averages):
        self.sampler1.sample(self.APD_buffer)
        self.APD_FORT_volts_science += self.APD_buffer[7]
        delay(1*ms)

    self.APD_FORT_volts_science /= self.APD_averages
    delay(100*ms)  # this is 100 ms to ensure the Luca takes the next shot

    if not self.MOT_light_off:
        self.dds_cooling_DP.sw.on()

    if self.do_PGC_in_MOT and self.t_PGC_in_MOT > 0:

        if not self.no_feedback:
            self.dds_FORT.set(frequency=self.f_FORT,
                              amplitude=self.stabilizer_FORT.amplitudes[1])

        self.zotino0.set_dac([self.AZ_bottom_volts_PGC, self.AZ_top_volts_PGC, self.AX_volts_PGC, self.AY_volts_PGC],
                             channels=self.coil_channels)

        self.dds_cooling_DP.set(frequency=self.f_cooling_DP_PGC,
                                amplitude=self.ampl_cooling_DP_MOT*self.p_cooling_DP_PGC)
        delay(self.t_PGC_in_MOT)

    self.dds_cooling_DP.sw.off()

@kernel
def first_shot(self):
    """
    non-chopped first atom readout

    warning: assumes the fiber AOMs are already on, which is usually the case
    :return:
    """
    if self.which_node != 'alice':  # edge counters only enabled on Alice gateware so far
        self.ttl_repump_switch.off()
        self.dds_cooling_DP.sw.on()
        t_gate_end = self.ttl_SPCM0.gate_rising(self.t_SPCM_first_shot)
        self.counts = self.ttl_SPCM0.count(t_gate_end)
        delay(0.1 * ms)
        self.dds_cooling_DP.sw.off()
    else:

        # if self.use_chopped_readout:
        #     ro_dma_handle = self.core_dma.get_handle("first_chopped_readout")
        #     delay(10 * ms)
        #
        #     # todo set RO coils here?
        #
        #     self.ttl_repump_switch.off()  # turns the RP AOM on
        #     self.dds_cooling_DP.sw.off()  # the chop sequence likes to turn the FORT off
        #
        #     delay(1 * ms)
        #     self.ttl7.pulse(100 * us)
        #     self.dds_FORT.sw.on()  # the chop sequence likes to turn the FORT off
        #
        #     delay(0.1 * ms)
        #
        #     # we want to initiate the chop playback and read in detector clicks while the chop sequence is playing.
        #     # the ttl.gate_rising(duration) function is equivalent to:
        #     #     ttl._set_sensitivity(1)
        #     #     delay(duration)
        #     #     ttl._set_sensitivity(0)
        #     #     return now_mu()
        #     #
        #     # we want the dma playback to happen during the gating, so we call the _set_sensitivity functions directly
        #
        #     self.ttl_SPCM0._set_sensitivity(1)
        #     self.core_dma.playback_handle(ro_dma_handle)
        #     self.ttl_SPCM0._set_sensitivity(0)
        #     self.counts = self.ttl_SPCM0.count(now_mu())
        #
        # else:
        self.ttl_repump_switch.off()
        self.dds_cooling_DP.sw.on()
        t_gate_end = self.ttl_SPCM0_counter.gate_rising(self.t_SPCM_first_shot)
        self.counts = self.ttl_SPCM0_counter.fetch_count()
        delay(0.1 * ms)
        self.dds_cooling_DP.sw.off()

@kernel
def second_shot(self):
    """
    non-chopped second atom readout

    warning: assumes the fiber AOMs are already on, which is usually the case
    :return:
    """
    if self.which_node != 'alice':  # edge counters only enabled on Alice gateware so far
        self.ttl_repump_switch.off()
        self.dds_cooling_DP.sw.on()
        t_gate_end = self.ttl_SPCM0.gate_rising(self.t_SPCM_second_shot)
        self.counts2 = self.ttl_SPCM0.count(t_gate_end)
        delay(0.1 * ms)
        self.dds_cooling_DP.sw.off()

    else:
        if self.use_chopped_readout:
            rtio_log("chop_RO_counter", 0)
            rtio_log("chop_RO_dma", 0)
            delay(10*ms)
            ro_dma_handle = self.core_dma.get_handle("second_chopped_readout")
            delay(10 * ms)

            # todo set RO coils here?

            self.ttl_repump_switch.off()  # turns the RP AOM on
            self.dds_cooling_DP.sw.off()  # the chop sequence likes to turn the FORT off

            delay(1 * ms)
            self.ttl7.pulse(100 * us)  # todo delete
            self.dds_FORT.sw.on()  # the chop sequence likes to turn the FORT off

            delay(10 * ms)

            # we want to initiate the chop playback and read in detector clicks while the chop sequence is playing.
            # the edge_counter.gate_rising(duration) function is equivalent to:
            #         edge_counter.set_config(
            #             count_rising=count_rising,
            #             count_falling=count_falling,
            #             send_count_event=False,
            #             reset_to_zero=True)
            #         delay_mu(duration_mu)
            #         edge_counter.set_config(
            #             count_rising=False,
            #             count_falling=False,
            #             send_count_event=True,
            #             reset_to_zero=False)
            #
            # we want the dma playback to happen during the gating, so we call the _set_sensitivity functions directly

            now = now_mu()
            rtio_log("chop_RO_counter", 1)
            self.ttl_SPCM0_counter.set_config(
                    count_rising=True,
                    count_falling=False,
                    send_count_event=False,
                    reset_to_zero=True)
            delay(1*us)
            rtio_log("chop_RO_dma", 1)
            self.core_dma.playback_handle(ro_dma_handle) # not sure if I need extra delay here
            rtio_log("chop_RO_dma", 0)
            at_mu(now + self.core.seconds_to_mu(self.t_SPCM_second_shot+10*us))
            self.ttl_SPCM0_counter.set_config(
                    count_rising=False,
                    count_falling=False,
                    send_count_event=True,
                    reset_to_zero=False)
            self.counts2 = self.ttl_SPCM0_counter.fetch_count()
            rtio_log("chop_RO_counter", 0)
            delay(10*ms)

        else:
            self.ttl_repump_switch.off()
            self.dds_cooling_DP.sw.on()
            t_gate_end = self.ttl_SPCM0_counter.gate_rising(self.t_SPCM_second_shot)
            self.counts2 = self.ttl_SPCM0_counter.fetch_count()
            delay(0.1 * ms)
            self.dds_cooling_DP.sw.off()

@kernel
def record_chopped_readout(self, readout_duration: TFloat, label: TStr):
    """

    :param self:
    :return:
    """
    # todo. copy OP chopping of the FORT but also chop the SPCM gate

    n_chop_cycles = int(readout_duration / self.t_RO_chop_period + 0.5)
    assert n_chop_cycles >= 1, "t_SPCM_first_shot should be > t_RO_chop_period"

    RO_pulse = self.t_RO_chop_period * self.duty_cycle_RO
    FORT_pulse = self.t_RO_chop_period * self.duty_cycle_FORT

    self.core.reset()

    start = now_mu()
    period_mu = self.core.seconds_to_mu(self.t_RO_chop_period)

    # does this need to be set within the dma context manager?
    RO_pulse_length_mu = self.core.seconds_to_mu(RO_pulse)
    FORT_pulse_length_mu = self.core.seconds_to_mu(FORT_pulse)
    FORT_on_mu = self.core.seconds_to_mu(0.0)
    RO_on_mu = self.core.seconds_to_mu(self.t_RO_chop_offset)
    gate_on_mu = self.core.seconds_to_mu(self.t_RO_gate_offset)

    with self.core_dma.record(label):

        # if not self.chopped_RO_light_off:
        for i in range(n_chop_cycles):
            at_mu(start + i * period_mu + FORT_on_mu)
            self.dds_FORT.sw.off()
            delay_mu(RO_pulse_length_mu)
            self.dds_FORT.sw.on()
            self.dds_cooling_DP.sw.on()

            at_mu(start + i * period_mu + gate_on_mu)
            self.ttl_SPCM_gate.off()  # unblocks the SPCM output
            at_mu(start + i * period_mu + RO_on_mu)
            self.dds_cooling_DP.sw.on()
            at_mu(start + i * period_mu + gate_on_mu + RO_pulse_length_mu)
            self.ttl_SPCM_gate.on()  # blocks the SPCM output
            at_mu(start + i * period_mu + RO_on_mu + RO_pulse_length_mu)
            self.dds_cooling_DP.sw.off()

            # cooling light doesn't seem synced up with SPCM gating based on photon count rate compared to RO_light_off case
            # with parallel:
            #     self.ttl_SPCM_gate.off() # unblocks the SPCM output
            #     self.dds_cooling_DP.sw.on()
            # delay_mu(RO_pulse_length_mu)
            # with parallel:
            #     self.dds_cooling_DP.sw.off()
            #     self.ttl_SPCM_gate.on()  # blocks the SPCM output
        # else:
        #     for i in range(n_chop_cycles):
        #         at_mu(start + i * period_mu + FORT_on_mu)
        #         self.dds_FORT.sw.off()
        #         delay_mu(RO_pulse_length_mu)
        #         self.dds_FORT.sw.on()
        #         at_mu(start + i * period_mu + gate_on_mu)
        #         self.ttl_SPCM_gate.off()  # unblocks the SPCM output
        #         delay_mu(RO_pulse_length_mu)
        #         self.ttl_SPCM_gate.on()  # blocks the SPCM output

@kernel
def record_chopped_blow_away(self):
    """

    :param self:
    :return:
    """

    # todo: change OP -> BA

    n_chop_cycles = int(self.t_blowaway/self.t_BA_chop_period + 0.5)
    # self.print_async("blowaway cycles:", n_chop_cycles)

    assert n_chop_cycles >= 1, "t_blowaway should be > t_BA_chop_period"

    BA_pulse = self.t_BA_chop_period * 0.35
    FORT_pulse = self.t_BA_chop_period - BA_pulse


    self.core.reset()

    with self.core_dma.record("chopped_blow_away"):

        start = now_mu()
        period_mu = self.core.seconds_to_mu(self.t_BA_chop_period)
        BA_pulse_length_mu = self.core.seconds_to_mu(BA_pulse)
        BA_on_mu = self.core.seconds_to_mu(FORT_pulse)
        FORT_pulse_length_mu = self.core.seconds_to_mu(FORT_pulse)
        FORT_on_mu = self.core.seconds_to_mu(BA_pulse)

        self.dds_FORT.sw.off()
        delay_mu(BA_pulse_length_mu)

        if not self.blowaway_light_off:
            self.dds_cooling_DP.sw.on()
        else:
            self.dds_cooling_DP.sw.off()

        for i in range(n_chop_cycles):
            at_mu(start+i*period_mu+FORT_on_mu)
            self.dds_FORT.sw.on()
            delay_mu(FORT_pulse_length_mu)
            self.dds_FORT.sw.off()
            at_mu(start+i*period_mu+BA_on_mu)
            # self.dds_cooling_DP.sw.on() # the cooling AOM seems incredibly slow so I'm just leaving it on the whole time
            delay_mu(BA_pulse_length_mu)
            # self.dds_cooling_DP.sw.off()
        self.dds_FORT.sw.on()

@kernel
def chopped_blow_away(self):

    ba_dma_handle = self.core_dma.get_handle("chopped_blow_away")

    self.ttl_repump_switch.on()  # turns off the RP AOM

    delay(0.1 * ms)

    # set coils for blowaway
    self.zotino0.set_dac(
        [self.AZ_bottom_volts_blowaway, self.AZ_top_volts_blowaway,
         self.AX_volts_blowaway, self.AY_volts_blowaway],
        channels=self.coil_channels)
    delay(0.1 * ms)

    with sequential:

        self.dds_cooling_DP.set(
            frequency=self.f_cooling_DP_blowaway,
            amplitude=self.ampl_cooling_DP_MOT)

        self.dds_AOM_A1.sw.off()
        self.dds_AOM_A2.sw.off()
        self.dds_AOM_A3.sw.off()
        self.dds_AOM_A4.sw.off()
        self.dds_AOM_A5.sw.off()

        if self.blowaway_light_off:
            self.dds_cooling_DP.sw.off()
            self.dds_AOM_A6.sw.off()
        else:
            # just turn the AOM up all the way. as long as we're 'saturating' the blowaway, it's okay if this doesn't
            # always give the same optical power
            self.dds_AOM_A6.set(frequency=self.AOM_A6_freq,
                                amplitude=dB_to_V(-7.0))
            self.dds_AOM_A6.sw.on()
            self.dds_cooling_DP.sw.on()

    self.core_dma.playback_handle(ba_dma_handle)

    # reset AOM RF powers
    self.dds_cooling_DP.sw.off()
    self.dds_cooling_DP.set(
        frequency=self.f_cooling_DP_RO,
        amplitude=self.ampl_cooling_DP_MOT)
    self.dds_AOM_A6.set(frequency=self.AOM_A6_freq,
                        amplitude=self.stabilizer_AOM_A6.amplitude)
    delay(0.1*ms)
    self.dds_AOM_A1.sw.on()
    self.dds_AOM_A2.sw.on()
    self.dds_AOM_A3.sw.on()
    self.dds_AOM_A4.sw.on()
    self.dds_AOM_A5.sw.on()
    self.dds_AOM_A6.sw.on()
    self.ttl_repump_switch.off()  # turns on the RP AOM

@kernel
def record_chopped_optical_pumping(self):
    """

    :param self:
    :return:
    """

    n_chop_cycles = int(self.t_pumping/self.t_OP_chop_period + 0.5)
    assert n_chop_cycles >= 1, "t_pumping should be > t_OP_chop_period"

    # todo: use duty cycle ExperimentVariables
    OP_pulse = self.t_OP_chop_period * self.duty_cycle_OP
    FORT_pulse = self.t_OP_chop_period * self.duty_cycle_FORT

    n_depump_chop_cycles = int(self.t_depumping /self.t_OP_chop_period)

    self.core.reset()

    with self.core_dma.record("chopped_optical_pumping"):

        # hardcoded for offsets for now, but this gives decent separation between the FORT and OP
        # pulses
        start = now_mu()
        period_mu = self.core.seconds_to_mu(self.t_OP_chop_period)

        OP_pulse_length_mu = self.core.seconds_to_mu(OP_pulse)
        FORT_pulse_length_mu = self.core.seconds_to_mu(FORT_pulse)
        FORT_on_mu = self.core.seconds_to_mu(0.0)
        OP_on_mu = self.core.seconds_to_mu(self.t_OP_chop_offset)

        if not (self.pumping_light_off or self.D1_off_in_OP_phase):
            for i in range(n_chop_cycles):
                at_mu(start+i*period_mu+FORT_on_mu)
                self.dds_FORT.sw.off()
                delay_mu(OP_pulse_length_mu)
                self.dds_FORT.sw.on()
                at_mu(start+i*period_mu+OP_on_mu)
                self.dds_D1_pumping_DP.sw.on()
                delay_mu(OP_pulse_length_mu)
                self.dds_D1_pumping_DP.sw.off()
        else:
            for i in range(n_chop_cycles):
                at_mu(start + i * period_mu + FORT_on_mu)
                self.dds_FORT.sw.off()
                delay_mu(OP_pulse_length_mu)
                self.dds_FORT.sw.on()

        if n_depump_chop_cycles > 0:

            # turn off the pumping repump
            self.dds_AOM_A5.sw.off()
            self.dds_AOM_A6.sw.off()
            self.dds_pumping_repump.sw.off()
            start = now_mu() + self.core.seconds_to_mu(0.5 * us)

            if not (self.pumping_light_off or self.D1_off_in_depump_phase):
                for i in range(n_depump_chop_cycles):
                    at_mu(start + i * period_mu + FORT_on_mu)
                    self.dds_FORT.sw.off()
                    delay_mu(OP_pulse_length_mu)
                    self.dds_FORT.sw.on()
                    at_mu(start + i * period_mu + OP_on_mu)
                    self.dds_D1_pumping_DP.sw.on()
                    delay_mu(OP_pulse_length_mu)
                    self.dds_D1_pumping_DP.sw.off()
            else:
                for i in range(n_depump_chop_cycles):
                    at_mu(start + i * period_mu + FORT_on_mu)
                    self.dds_FORT.sw.off()
                    delay_mu(OP_pulse_length_mu)
                    self.dds_FORT.sw.on()

@kernel
def chopped_optical_pumping(self):
    """

    :param self:
    :return:
    """

    op_dma_handle = self.core_dma.get_handle("chopped_optical_pumping")
    if self.t_depumping + self.t_pumping > 3*ms:
        delay(2000 * us)  # we need extra slack
    self.ttl_repump_switch.on()  # turns off the MOT RP AOM
    self.dds_cooling_DP.sw.off()  # no cooling light

    # make sure the fiber AOMs are on for delivery of the pumping repump
    if not self.pumping_light_off:
        self.dds_pumping_repump.sw.on()

    # self.dds_AOM_A5.set(frequency=self.AOM_A5_freq,amplitude=dB_to_V(self.p_pumping_repump_A5))
    # self.dds_AOM_A6.set(frequency=self.AOM_A6_freq,amplitude=dB_to_V(self.p_pumping_repump_A6))

    self.dds_AOM_A5.sw.on()
    self.dds_AOM_A6.sw.on()

    delay(1*ms)


    self.dds_excitation.sw.on()
    self.ttl_excitation_switch.off()

    # set coils for pumping
    self.zotino0.set_dac(
        [self.AZ_bottom_volts_OP, self.AZ_top_volts_OP, self.AX_volts_OP, self.AY_volts_OP],
        channels=self.coil_channels)
    delay(0.4 * ms)  # coil relaxation time

    with sequential:

        delay(1*us)

        self.core_dma.playback_handle(op_dma_handle)
        delay(self.t_depumping)

        self.dds_D1_pumping_DP.sw.off()
        self.dds_pumping_repump.sw.off()

    delay(2*us)

    self.dds_AOM_A5.sw.off()
    self.dds_AOM_A6.sw.off()

    delay(100 * us)

    # self.dds_AOM_A5.set(frequency=self.AOM_A5_freq, amplitude=self.stabilizer_AOM_A5.amplitude)
    # self.dds_AOM_A6.set(frequency=self.AOM_A6_freq, amplitude=self.stabilizer_AOM_A6.amplitude)


    delay(1*ms)
    self.dds_excitation.sw.off()
    self.ttl_excitation_switch.on()

    # reset MOT power
    self.dds_cooling_DP.sw.off()

@kernel
def measure_FORT_MM_fiber(self):
    measurement_buf = np.array([0.0]*8)
    measurement = 0.0
    avgs = 50
    for i in range(avgs):
        self.sampler1.sample(measurement_buf)
        measurement += measurement_buf[self.FORT_MM_sampler_ch]
        delay(0.1*ms)
    measurement /= avgs
    self.append_to_dataset("FORT_MM_science_volts", measurement)

@kernel
def measure_GRIN1(self):

    """
    used for monitring GRIN1
    GRIN1_sampler_ch = 4 defined at "BaseExperiment.py"

    this is used in chopped_optical_pumping

    can be used for monitoring exitation power

    """
    measurement_buf = np.array([0.0]*8)
    measurement = 0.0
    avgs = 50

    self.dds_FORT.sw.off()
    self.ttl_repump_switch.on()  # turns the RP AOM off
    self.dds_cooling_DP.sw.off()

    #turning D1 on
    # todo: erase this line if doing feedback for D1
    self.dds_excitation.set(frequency=self.f_excitation, amplitude=dB_to_V(0.0))

    self.dds_D1_pumping_DP.sw.on()
    self.dds_excitation.sw.on()
    self.ttl_excitation_switch.off()

    delay(0.1*ms)

    for i in range(avgs):
        self.sampler1.sample(measurement_buf)
        measurement += measurement_buf[self.GRIN1_sampler_ch]
        delay(0.1*ms)
    measurement /= avgs
    self.append_to_dataset("GRIN1_D1_monitor", measurement)

    # turning D1 off & repump on
    self.dds_excitation.set(frequency=self.f_excitation, amplitude=self.stabilizer_excitation.amplitudes[0])
    self.dds_D1_pumping_DP.sw.off()
    self.ttl_repump_switch.off()  # turns the RP AOM on

    delay(0.1*ms)

    for i in range(avgs):
        self.sampler1.sample(measurement_buf)
        measurement += measurement_buf[self.GRIN1_sampler_ch]
        delay(0.1*ms)
    measurement /= avgs
    self.append_to_dataset("GRIN1_EXC_monitor", measurement)


    # turning EXC OFF
    self.dds_excitation.sw.off()
    self.ttl_excitation_switch.on()

    delay(0.1 * ms)

@kernel
def measure_REPUMP(self):
    """
    used for monitring PUMPING REPUMP power

    This is in end_measurement

    AOM1: Sampler0, 7
    AOM2: Sampler0, 5

    """
    measurement_buf = np.array([0.0]*8)
    measurement1 = 0.0 # Repump 1
    measurement2 = 0.0 # Repump 2

    avgs = 50

    # ttl7 - trigger

    self.dds_FORT.sw.off()
    self.ttl_repump_switch.off()  # turns the RP AOM on
    self.dds_pumping_repump.sw.off()
    self.dds_cooling_DP.sw.off()

    delay(0.1 * ms)

    self.dds_AOM_A1.sw.on()
    self.dds_AOM_A2.sw.on()

    delay(0.1 * ms)

    # Repump 1
    for i in range(avgs):
        self.sampler0.sample(measurement_buf)
        delay(0.1 * ms)
        measurement1 += measurement_buf[7] # Repump 1
        delay(0.1 * ms)
        measurement2 += measurement_buf[5] # Repump 2


    measurement1 /= avgs
    measurement2 /= avgs

    self.append_to_dataset("REPUMP1_monitor", measurement1)
    self.append_to_dataset("REPUMP2_monitor", measurement2)

    self.ttl_repump_switch.on()  # turns the RP AOM off
    self.dds_AOM_A1.sw.off()
    self.dds_AOM_A2.sw.off()

    delay(0.1 * ms)

@kernel
def measure_PUMPING_REPUMP(self):
    """
    used for monitring REPUMP power
    PUMPING_REPUMP1_monitor, PUMPING_REPUMP2_monitor defined

    This is in end_measurement

    Pumping Repumper is sent to AOM5 & 6

    AOM5: Sampler0, 1
    AOM6: Sampler0, 2

    """
    measurement_buf = np.array([0.0]*8)
    measurement1 = 0.0 # Repump 5
    measurement2 = 0.0 # Repump 6

    avgs = 50

    self.dds_FORT.sw.off()
    self.ttl_repump_switch.on()  # turns the RP AOM off
    self.dds_cooling_DP.sw.off()

    self.dds_pumping_repump.sw.on() # turns of PR AOM

    # self.dds_AOM_A5.set(frequency=self.AOM_A5_freq,amplitude=dB_to_V(-8.0))
    # self.dds_AOM_A6.set(frequency=self.AOM_A6_freq,amplitude=dB_to_V(-8.0))

    self.dds_AOM_A5.sw.on()
    self.dds_AOM_A6.sw.on()

    delay(0.1 * ms)

    # Repump 1
    for i in range(avgs):
        self.sampler0.sample(measurement_buf)
        delay(0.1 * ms)
        measurement1 += measurement_buf[1] # Repump 5
        delay(0.1 * ms)
        measurement2 += measurement_buf[2] # Repump 6


    measurement1 /= avgs
    measurement2 /= avgs

    self.append_to_dataset("PUMPING_REPUMP1_monitor", measurement1)
    self.append_to_dataset("PUMPING_REPUMP2_monitor", measurement2)

    self.dds_pumping_repump.sw.off()  # turns the PR AOM off

    # self.dds_AOM_A5.set(frequency=self.AOM_A5_freq, amplitude=self.stabilizer_AOM_A5.amplitude)
    # self.dds_AOM_A6.set(frequency=self.AOM_A6_freq, amplitude=self.stabilizer_AOM_A6.amplitude)


    self.dds_AOM_A5.sw.off()
    self.dds_AOM_A6.sw.off()

    delay(0.1 * ms)

@kernel
def measure_MOT_end(self):
    """
    used for monitring MOT power
    MOT_end_monitor1 defined

    This is in end_measurement

    AOM1: Sampler0, 7
    AOM2: Sampler0, 5
    AOM3: Sampler0, 3
    AOM4: Sampler0, 4
    AOM5: Sampler0, 1
    AOM6: Sampler0, 2

    """
    ao_s1 = 7
    ao_s2 = 5
    ao_s3 = 3
    ao_s4 = 4
    ao_s5 = 1
    ao_s6 = 2

    avgs = 50


    delay(0.1 * ms)

    self.dds_FORT.sw.off()
    self.ttl_repump_switch.on()  # turns the RP AOM on
    self.dds_cooling_DP.sw.on()

    # self.dds_AOM_A1.sw.off()
    # self.dds_AOM_A2.sw.off()
    # self.dds_AOM_A3.sw.off()
    # self.dds_AOM_A4.sw.off()
    # self.dds_AOM_A5.sw.off()
    # self.dds_AOM_A6.sw.off()

    ### MOT1 & MOT2
    measurement_buf = np.array([0.0] * 8)
    measurement1 = 0.0  # MOT1
    measurement2 = 0.0  # MOT2
    measurement3 = 0.0  # MOT5

    self.dds_AOM_A1.sw.on()
    self.dds_AOM_A2.sw.on()
    self.dds_AOM_A5.sw.on()

    delay(0.1 * ms)

    for i in range(avgs):
        self.sampler0.sample(measurement_buf)
        delay(0.1 * ms)
        measurement1 += measurement_buf[ao_s1]  # MOT1
        delay(0.1 * ms)
        measurement2 += measurement_buf[ao_s2]  # MOT2
        measurement3 += measurement_buf[ao_s5]  # MOT5

    measurement1 /= avgs
    measurement2 /= avgs
    measurement3 /= avgs

    self.append_to_dataset("MOT1_end_monitor", measurement1)
    self.append_to_dataset("MOT2_end_monitor", measurement2)
    self.append_to_dataset("MOT5_end_monitor", measurement3)

    self.dds_AOM_A1.sw.off()
    self.dds_AOM_A2.sw.off()
    self.dds_AOM_A5.sw.off()

    delay(0.1 * ms)


    ### MOT3 & MOT4

    self.dds_AOM_A3.sw.on()
    self.dds_AOM_A4.sw.on()
    self.dds_AOM_A6.sw.on()

    measurement_buf = np.array([0.0] * 8)
    measurement1 = 0.0
    measurement2 = 0.0
    measurement3 = 0.0

    delay(0.1 * ms)

    for i in range(avgs):
        self.sampler0.sample(measurement_buf)
        delay(0.1 * ms)
        measurement1 += measurement_buf[ao_s3]
        delay(0.1 * ms)
        measurement2 += measurement_buf[ao_s4]
        measurement3 += measurement_buf[ao_s6]

    measurement1 /= avgs
    measurement2 /= avgs
    measurement3 /= avgs

    self.append_to_dataset("MOT3_end_monitor", measurement1)
    self.append_to_dataset("MOT4_end_monitor", measurement2)
    self.append_to_dataset("MOT6_end_monitor", measurement3)

    self.dds_AOM_A3.sw.off()
    self.dds_AOM_A4.sw.off()
    self.dds_AOM_A6.sw.off()

    delay(0.1 * ms)


    ### MOT5 & MOT6

    self.dds_AOM_A5.sw.on()
    self.dds_AOM_A6.sw.on()

    measurement_buf = np.array([0.0] * 8)
    measurement1 = 0.0
    measurement2 = 0.0

    delay(0.1 * ms)

    for i in range(avgs):
        self.sampler0.sample(measurement_buf)
        delay(0.1 * ms)
        measurement1 += measurement_buf[ao_s5]
        delay(0.1 * ms)
        measurement2 += measurement_buf[ao_s6]

    measurement1 /= avgs
    measurement2 /= avgs

    self.append_to_dataset("MOT5_end_monitor", measurement1)
    self.append_to_dataset("MOT6_end_monitor", measurement2)

    self.dds_AOM_A5.sw.off()
    self.dds_AOM_A6.sw.off()

    delay(0.1 * ms)



@kernel
def end_measurement(self):
    """
    End the measurement by setting datasets and deciding whether to increment the measuement index
    :param self:
    :return measurement: TInt32, the measurement index
    """

    # update the datasets
    if not self.no_first_shot:
        self.append_to_dataset('photocounts_current_iteration', self.counts)
        self.counts_list[self.measurement] = self.counts

    # update the datasets
    self.set_dataset(self.measurements_progress, 100*self.measurement/self.n_measurements, broadcast=True)
    self.append_to_dataset('photocounts2_current_iteration', self.counts2)
    self.counts2_list[self.measurement] = self.counts2
    self.append_to_dataset("photocounts_FORT_science", self.counts_FORT_science)

    delay(1*ms)
    measure_FORT_MM_fiber(self)
    delay(1*ms)
    measure_REPUMP(self)
    delay(1*ms)
    measure_GRIN1(self)
    delay(1*ms)
    measure_PUMPING_REPUMP(self)
    delay(1*ms)
    measure_MOT_end(self)
    delay(1*ms)

    advance = 1
    if self.__class__.__name__ != 'ExperimentCycler':
        if self.require_atom_loading_to_advance:
            if not self.counts/self.t_SPCM_first_shot > self.single_atom_counts_per_s:
                advance *= 0
        if self.require_D1_lock_to_advance:
            self.ttl_D1_lock_monitor.sample_input()
            delay(0.1 * ms)
            laser_locked = int(1 - self.ttl_D1_lock_monitor.sample_get())
            advance *= laser_locked
            if not laser_locked:
                logging.warning("D1 laser not locked")

    if advance:
        self.measurement += 1
        if not self.no_first_shot:
            self.append_to_dataset('photocounts', self.counts)
        self.append_to_dataset('photocounts2', self.counts2)

@rpc(flags={"async"})
def set_RigolDG1022Z(frequency: TFloat, vpp: TFloat, vdc: TFloat):
    """
    Set the frequency, V_DC, and V_pp of a connected Rigol DG1000 series function generator.

    See https://www.batronix.com/pdf/Rigol/ProgrammingGuide/DG1000Z_ProgrammingGuide_EN.pdf

    :return TFloat: the frequency of the Rigol DG1022Z in Hz
    """
    # NAME = 'USB0::0x1AB1::0x0642::DG1ZA252402452::INSTR'

    rm = visa.ResourceManager()
    instruments = rm.list_resources()
    for instr in instruments:
        if 'DG' in instr:
            NAME = instr
            break

    funcgen = rm.open_resource(NAME, timeout=20, chunk_size=1024000)
    funcgen.timeout = 1000  # in ms I think

    # Make sure mod is off
    funcgen.write(r":SOUR1:MOD OFF")

    # Set the frequency
    freq = int(frequency)
    funcgen.write(f":SOUR1:FREQ {freq}")
    actual_freq = float(funcgen.query("SOUR1:FREQ?"))
    assert actual_freq == freq, "Oops! The device frequency is not set to the requested value!"
    print(f"f_carrier: {actual_freq} Hz")

    # Set the Vpp
    funcgen.write(f":SOUR1:VOLT {vpp}")
    actual_vpp = float(funcgen.query("SOUR1:VOLT?"))
    assert actual_vpp == vpp, "Oops! The device V_pp is not set to the requested value!"
    print(f"Vpp: {actual_vpp} V")

    # Set the dc offset
    funcgen.write(f":SOUR1:VOLT:OFFS {vdc}")
    actual_vdc = float(funcgen.query("SOUR1:VOLT:OFFS?"))
    assert actual_vdc == vdc, "Oops! The device V_DC is not set to the requested value!"
    print(f"Vdc: {actual_vdc} V")

###############################################################################
# 2. EXPERIMENT FUNCTIONS
# These are the experiments we run, and the name of each should end with
# experiment in order to have it show up in GeneralVariableScans
###############################################################################

@kernel
def test_experiment(self):
    """
    self is the experiment instance to which ExperimentVariables are bound
    """
    x = self.t_blowaway
    self.append_to_dataset('test_dataset', x)
    self.print_async(x)

@kernel
def MOT_loading_experiment(self):
    self.core.reset()

    for measurement in range(self.n_measurements):
        self.laser_stabilizer.run()  # this tunes the MOT and FORT AOMs
        load_MOT_and_FORT(self)

@kernel
def atom_loading_experiment(self):
    """
    The most basic two-readout single atom experiment.

    Load a MOT, load a single atom, readout, wait self.t_delay_between_shots, readout again.

    :param self: an experiment instance.
    :return:
    """

    self.core.reset()

    self.counts = 0
    self.counts2 = 0
    rtio_log("2nd_shot_block", 0)

    if self.use_chopped_readout:
        # record_chopped_readout(self, self.t_SPCM_first_shot, "first_chopped_readout")
        # delay(100*ms)
        record_chopped_readout(self, readout_duration=self.t_SPCM_second_shot, label="second_chopped_readout")
        delay(100*ms)

    self.print_async("in exp:",self.AZ_bottom_volts_MOT, self.AZ_top_volts_MOT, self.AX_volts_MOT, self.AY_volts_MOT)

    self.require_D1_lock_to_advance = False # override experiment variable

    self.set_dataset(self.count_rate_dataset,
                     [0.0],
                     broadcast=True)

    self.measurement = 0
    while self.measurement < self.n_measurements:

        if self.enable_laser_feedback:
            self.laser_stabilizer.run()  # this tunes the MOT and FORT AOMs

        load_MOT_and_FORT(self)

        delay(0.1*ms)
        self.zotino0.set_dac(
            [self.AZ_bottom_volts_RO, self.AZ_top_volts_RO, self.AX_volts_RO, self.AY_volts_RO],
            channels=self.coil_channels)

        # set the cooling DP AOM to the readout settings
        self.dds_cooling_DP.set(frequency=self.f_cooling_DP_RO,
                                amplitude=self.ampl_cooling_DP_MOT*self.p_cooling_DP_RO)

        if not self.no_first_shot:
            first_shot(self)

        if self.t_FORT_drop > 0:
            self.dds_FORT.sw.off()
            delay(self.t_FORT_drop)
            self.dds_FORT.sw.on()

        self.dds_cooling_DP.sw.off()
        self.ttl_SPCM_gate.pulse(self.t_delay_between_shots) # blocks the SPCM
        self.dds_cooling_DP.sw.on()

        rtio_log("2nd_shot_block",1)
        second_shot(self)
        rtio_log("2nd_shot_block", 0)

        self.dds_AOM_A1.sw.off()
        self.dds_AOM_A2.sw.off()
        self.dds_AOM_A3.sw.off()
        self.dds_AOM_A4.sw.off()
        self.dds_AOM_A5.sw.off()
        self.dds_AOM_A6.sw.off()

        end_measurement(self)

    self.dds_FORT.sw.off()

@kernel
def trap_frequency_experiment(self):
    """
    For spectroscopy of the trap vibrational frequencies with the Rigol D11022Z function generator.

    One way to use this is with GeneralVariableScan and scan_variable1 = f_Rigol_modulation,
    where f_Rigol_modulation must be within the boundaries Rigol_carrier_frequency +/- Rigol_FM_deviation
    specified in ExpermientVariables (which in turn must be set on the Rigol D11022Z).

    :param self: an experiment instance.
    :return:
    """

    set_RigolDG1022Z(frequency=self.Rigol_carrier_frequency,
                     vpp=self.Rigol_V_pp,
                     vdc=self.Rigol_V_DC)

    self.core.reset()

    self.counts = 0
    self.counts2 = 0

    self.require_D1_lock_to_advance = False # override experiment variable

    self.set_dataset(self.count_rate_dataset,
                     [0.0],
                     broadcast=True)

    self.measurement = 0
    while self.measurement < self.n_measurements:

        #TODO: just set the rigol frequency using pyvisa

        if self.enable_laser_feedback:
            self.laser_stabilizer.run()  # this tunes the MOT and FORT AOMs

        load_MOT_and_FORT(self)

        delay(0.1*ms)
        self.zotino0.set_dac(
            [self.AZ_bottom_volts_RO, self.AZ_top_volts_RO, self.AX_volts_RO, self.AY_volts_RO],
            channels=self.coil_channels)

        # set the FORT AOM to the science setting. this is only valid if we have run
        # feedback to reach the corresponding setpoint first, which in this case, happened in load_MOT_and_FORT
        self.dds_FORT.set(frequency=self.f_FORT,
                                amplitude=self.stabilizer_FORT.amplitudes[1])

        # set the cooling DP AOM to the readout settings
        self.dds_cooling_DP.set(frequency=self.f_cooling_DP_RO,
                                amplitude=self.ampl_cooling_DP_MOT*self.p_cooling_DP_RO)

        if not self.no_first_shot:
            first_shot(self)

        ##############################################################################
        # modulate the FORT
        ##############################################################################

        self.FORT_mod_switch.on()  # toggle the modulation to the VCA
        delay(10 * ms)
        self.FORT_mod_switch.off()
        delay(1 * ms)

        second_shot(self)

        end_measurement(self)

    self.dds_FORT.sw.off()

@kernel
def microwave_Rabi_experiment(self):
    """
    Microwave and optical pumping experiment.
    This experiment is used for any experiment which involves optical pumping and a microwave rotations with a constant
    microwave amplitude over a specified (possibly zero) duration. For example, this can be used for
    - microwave Rabi oscillations to test the optical pumping fidelity
    - microwave spectroscopy (useful for zeroing the magnetic field by finding the resonances for different ground state
    transitions |F=1,m>->|F=2,m'>)
    - depumping measurements (by using a non-zero depump time)

    self is the experiment instance to which ExperimentVariables are bound
    """

    self.core.reset()

    self.counts = 0
    self.counts2 = 0

    self.set_dataset(self.count_rate_dataset,
                     [0.0],
                     broadcast=True)

    if self.t_pumping > 0.0:
        record_chopped_optical_pumping(self)
        delay(100*ms)
    if self.t_blowaway > 0.0:
        record_chopped_blow_away(self)
        delay(100*ms)

    delay(10 * ms)
    self.dds_microwaves.set(frequency=self.f_microwaves_dds, amplitude=dB_to_V(self.p_microwaves))
    delay(10 * ms)
    self.dds_microwaves.sw.on()
    delay(100 * ms)

    self.measurement = 0
    while self.measurement < self.n_measurements:

        if self.enable_laser_feedback:
            self.laser_stabilizer.run()  # this tunes the MOT and FORT AOMs

            # bug -- microwave dds is off after AOM feedback; not clear why yet. for now, just turn it back on
            self.dds_microwaves.sw.on()

        load_MOT_and_FORT(self)

        delay(0.1 * ms)
        self.zotino0.set_dac(
            [self.AZ_bottom_volts_RO, self.AZ_top_volts_RO, self.AX_volts_RO, self.AY_volts_RO],
            channels=self.coil_channels)

        # set the FORT AOM to the readout settings
        self.dds_FORT.set(frequency=self.f_FORT,
                          amplitude=self.stabilizer_FORT.amplitudes[1])

        # set the cooling DP AOM to the readout settings
        self.dds_cooling_DP.set(frequency=self.f_cooling_DP_RO,
                                amplitude=self.ampl_cooling_DP_MOT * self.p_cooling_DP_RO)

        if not self.no_first_shot:
            first_shot(self)
        delay(1 * ms) # leave the repump on so atoms are left in F=2
        self.ttl_repump_switch.on()  # turns the RP AOM off

        if self.t_FORT_drop > 0:
            self.dds_FORT.sw.off()
            delay(self.t_FORT_drop)
            self.dds_FORT.sw.on()

        ############################
        # optical pumping phase - pumps atoms into F=1,m_F=0
        ############################
        if self.t_pumping > 0.0:
            chopped_optical_pumping(self)
            delay(1*ms)

        ############################
        # microwave phase
        ############################

        if self.t_microwave_pulse > 0.0:
            # self.ttl_repump_switch.on()  # turns off the RP AOM

            # todo: set coils for microwaves. good for diagnostics-- we can use this phase to zero the B-field
            self.zotino0.set_dac(
                [self.AZ_bottom_volts_OP, self.AZ_top_volts_OP,
                 self.AX_volts_OP, self.AY_volts_OP],
                channels=self.coil_channels)
            delay(0.1*ms)
            # self.zotino0.set_dac(
            #     [self.AZ_bottom_volts_microwave, self.AZ_top_volts_microwave,
            #      self.AX_volts_microwave, self.AY_volts_microwave],
            #     channels=self.coil_channels)
            # delay(0.1*ms)

            self.ttl7.pulse(self.t_exp_trigger)  # in case we want to look at signals on an oscilloscope

            self.ttl_microwave_switch.off()

            delay(self.t_microwave_pulse)

            self.ttl_microwave_switch.on()

        ############################
        # blow-away phase - push out atoms in F=2 only
        ############################

        if self.t_blowaway > 0.0:
            chopped_blow_away(self)

        # set the FORT AOM to the readout settings
        self.dds_FORT.set(frequency=self.f_FORT,
                          amplitude=self.stabilizer_FORT.amplitudes[1])

        # take the second shot
        self.zotino0.set_dac(
            [self.AZ_bottom_volts_RO, self.AZ_top_volts_RO, self.AX_volts_RO, self.AY_volts_RO],
            channels=self.coil_channels)
        delay(0.1 * ms)

        second_shot(self)

        self.dds_AOM_A1.sw.off()
        self.dds_AOM_A2.sw.off()
        self.dds_AOM_A3.sw.off()
        self.dds_AOM_A4.sw.off()
        self.dds_AOM_A5.sw.off()
        self.dds_AOM_A6.sw.off()

        end_measurement(self)
        delay(5 * ms) ### hopefully to avoid underflow.

    delay(10*ms)
    self.dds_FORT.sw.off()
    delay(1*ms)
    self.dds_microwaves.sw.off()

@kernel
def single_photon_experiment(self):
    """
    This experiment pumps the atom into F=1,m=0 then excites it to F=0,0.
    This sequence is repeated multiple times, but only one SPCM is monitored,
    so we can not use this result to verify single photons. We can use it to
    make sure we are only getting one or zero clicks after each excitation
    attempt, and that the one click events only occur when there is an atom
    loaded.

    self is the experiment instance to which ExperimentVariables are bound
    """

    self.core.reset()

    # overwritten below but initialized here so they are always initialized
    self.counts = 0
    self.counts2 = 0
    excitation_counts = 0
    excitation_counts_array = [0]
    rtio_log("2nd_shot_block", 0) # todo: delete. for debugging.

    self.set_dataset(self.count_rate_dataset,
                     [0.0],
                     broadcast=True)

    record_chopped_optical_pumping(self)
    delay(100*ms)

    if self.verify_OP_in_photon_experiment:
        if self.t_blowaway > 0.0:
            record_chopped_blow_away(self)
            delay(100*ms)

        self.dds_microwaves.set(frequency=self.f_microwaves_dds, amplitude=dB_to_V(self.p_microwaves))
        delay(10 * ms)
        self.dds_microwaves.sw.on()
        delay(100 * ms)

    op_dma_handle = self.core_dma.get_handle("chopped_optical_pumping")

    self.measurement = 0
    while self.measurement < self.n_measurements:

        excitation_counts_array = [0] * self.n_excitation_cycles
        excitation_counts_array1 = [0] * self.n_excitation_cycles

        if self.enable_laser_feedback:
            self.laser_stabilizer.run()  # this tunes the MOT and FORT AOMs

            # bug -- microwave dds is off after AOM feedback; not clear why yet. for now, just turn it back on
            self.dds_microwaves.sw.on()

        delay(10*ms)
        load_MOT_and_FORT(self)

        delay(0.1 * ms)
        self.zotino0.set_dac(
            [self.AZ_bottom_volts_RO, self.AZ_top_volts_RO, self.AX_volts_RO, self.AY_volts_RO],
            channels=self.coil_channels)

        # set the FORT AOM to the readout settings
        self.dds_FORT.set(frequency=self.f_FORT,
                          amplitude=self.stabilizer_FORT.amplitudes[1])

        # set the cooling DP AOM to the readout settings
        self.dds_cooling_DP.set(frequency=self.f_cooling_DP_RO,
                                amplitude=self.ampl_cooling_DP_MOT * self.p_cooling_DP_RO)

        if not self.no_first_shot:
            first_shot(self)
        delay(1 * ms)
        self.ttl_repump_switch.on()  # turns the RP AOM off

        if self.t_FORT_drop > 0:
            self.dds_FORT.sw.off()
            delay(self.t_FORT_drop)
            self.dds_FORT.sw.on()

        # self.ttl7.pulse(10 * us)  # in case we want to look at signals on an oscilloscope

        ########################################################
        # lower level optical pumping and excitation sequence to optimize for speed
        ########################################################

        excitation_counts = 0
        excitation_counts1 = 0

        self.ttl_SPCM_gate.on()  # blocks the SPCM output - this is related to the atom readouts undercounting
        loop_start_mu = now_mu()

        self.zotino0.set_dac(
            [self.AZ_bottom_volts_OP, self.AZ_top_volts_OP, self.AX_volts_OP, self.AY_volts_OP],
            channels=self.coil_channels)
        delay(0.4 * ms)  # coil relaxation time

        # this will stay on for the entire excition + OP loop, because both the D1 and excitation light use it
        self.dds_excitation.sw.on()

        # don't use gate_rising. set the sensitivity, do the pumping and excitation sequence, and count the photons
        # after excitation attempts. we'll block the counter channel with an external switch so we don't get any clicks
        # during the OP phases.
        self.ttl_SPCM0._set_sensitivity(1)
        self.ttl_SPCM1._set_sensitivity(1)
        for excitaton_cycle in range(self.n_excitation_cycles): # todo: revert later

            delay(0.5*ms)

            # low level pumping sequnce is more time efficient than the prepackaged chopped_optical_pumping function.
            # todo: make sure this is consistent with any updates in chopped_optical_pumping function
            ############################
            # optical pumping phase - pumps atoms into F=1,m_F=0
            ############################


            if self.t_pumping > 0.0:
                self.dds_excitation.set(frequency=self.f_excitation, amplitude=dB_to_V(0.0))
                self.ttl_repump_switch.on()  # turns off the MOT RP AOM

                if not self.pumping_light_off:
                    self.dds_pumping_repump.sw.on()

                self.dds_AOM_A5.sw.on()
                self.dds_AOM_A6.sw.on()

                delay(.1 * ms) # maybe this can be even shorter

                self.ttl_excitation_switch.off()

                with sequential:

                    delay(1 * us)

                    self.core_dma.playback_handle(op_dma_handle)
                    delay(self.t_depumping)

                    self.dds_D1_pumping_DP.sw.off()
                    self.dds_pumping_repump.sw.off() # turn the repump back on

                delay(2 * us)
                self.dds_AOM_A5.sw.off()
                self.dds_AOM_A6.sw.off()

                self.ttl_excitation_switch.on()

                ############################
                # microwave phase - ONLY USED FOR VERIFYING OP.
                ############################

                if self.t_microwave_pulse > 0.0 and self.verify_OP_in_photon_experiment:
                    self.ttl_microwave_switch.off()
                    delay(self.t_microwave_pulse)
                    self.ttl_microwave_switch.on()
                    delay(0.1*ms)

                ############################
                # blow-away phase - push out atoms in F=2 only
                ############################

                if self.t_blowaway > 0.0 and self.verify_OP_in_photon_experiment:
                    chopped_blow_away(self)

            ############################
            # excitation phase - excite F=1,m=0 -> F'=0,m'=0, detect photon
            ############################
            self.dds_excitation.set(frequency=self.f_excitation, amplitude=self.stabilizer_excitation.amplitudes[0])
            self.ttl_repump_switch.off()  # repump AOM is on for excitation

            now = now_mu()

            # messy and confusing. todo: try to improve this
            # at_mu(now+10)
            # self.ttl_repump_switch.off()  # repump AOM is on for excitation
            # mu_offset = 800 # accounts for various latencies
            # at_mu(now + mu_offset - 200) # make sure stuff is off, no more Raman photons from FORT
            # at_mu(now + mu_offset+100+self.gate_start_offset_mu) # allow for repump rise time and FORT after-pulsing
            # t_collect = now_mu()
            # t_excite = now_mu()
            # pulses_over_mu = 0
            # for attempt in range(self.n_excitation_attempts):
            #     at_mu(now + mu_offset + 201 + int(attempt * (self.t_excitation_pulse / ns + 100))
            #           + self.gate_start_offset_mu)
            #     self.dds_excitation.sw.pulse(self.t_excitation_pulse)
            #     at_mu(now + mu_offset + 741 + int(attempt * (self.t_excitation_pulse / ns + 100) -
            #                                       0.1*self.t_excitation_pulse / ns) +self.gate_start_offset_mu)
            #
            #     # fast switch to gate SPCM output - why am I using an external switch
            #     # instead of the gate input on the SPCM?
            #     self.ttl_SPCM_gate.off()
            #     delay(0.2*self.t_excitation_pulse+100*ns + self.gate_switch_offset)
            #     self.ttl_SPCM_gate.on()
            #
            #     pulses_over_mu = now_mu() # overwrite each loop iteration

            self.dds_FORT.sw.off()
            at_mu(now+150)
            self.ttl_excitation_switch.off()
            at_mu(now + 150 + self.gate_start_offset_mu)
            self.ttl_SPCM_gate.off()
            at_mu(now + 150 + int(self.t_excitation_pulse/ns))
            self.ttl_excitation_switch.on()
            # delay(self.t_photon_collection_time - self.t_excitation_pulse)
            at_mu(now + 150 + int(self.t_excitation_pulse / ns) + int((self.t_photon_collection_time - self.t_excitation_pulse)/ns) + self.gate_start_offset_mu)
            self.ttl_SPCM_gate.on()

            at_mu(now + 150 + int(self.t_excitation_pulse / ns) + int((self.t_photon_collection_time - self.t_excitation_pulse)/ns))
            self.dds_FORT.sw.on()
            pulses_over_mu = now_mu()

            delay(1*us)
            self.ttl_repump_switch.on() # block MOT repump (and therefore also the excitation light)



            # todo: terrible way of doing this. set the sensitivity of the gate at the beginning of the loop.
            #  it will only register events when the SPCM switch lets events through.
            excitation_counts = self.ttl_SPCM0.count(
                pulses_over_mu)  # this is the number of clicks we got over n_excitation attempts
            excitation_counts1 = self.ttl_SPCM1.count(
                pulses_over_mu)  # this is the number of clicks we got over n_excitation attempts
            excitation_counts_array[excitaton_cycle] = excitation_counts
            excitation_counts_array1[excitaton_cycle] = excitation_counts1
            delay(0.1*ms) # ttl count consumes all the RTIO slack.
            # loop_over_mu = now_mu()

            # todo: delete
            # self.print_async("bottom of excitation loop",now_mu() - loop_start_mu)

            ############################
            # recooling phase
            ############################

            # # todo: use a specific detuning for this stage?
            delay(1*ms)
            if self.t_recooling > 0:
                self.dds_cooling_DP.sw.on()
                self.ttl_repump_switch.off()
                self.dds_AOM_A1.sw.on()
                self.dds_AOM_A2.sw.on()
                self.dds_AOM_A3.sw.on()
                self.dds_AOM_A4.sw.on()
                self.dds_AOM_A5.sw.on()
                self.dds_AOM_A6.sw.on()

                delay(self.t_recooling)

                self.dds_cooling_DP.sw.off()
                self.ttl_repump_switch.on()
                self.dds_AOM_A1.sw.off()
                self.dds_AOM_A2.sw.off()
                self.dds_AOM_A3.sw.off()
                self.dds_AOM_A4.sw.off()
                self.dds_AOM_A5.sw.off()
                self.dds_AOM_A6.sw.off()
                delay(1*ms)


        self.ttl_SPCM0._set_sensitivity(0) # close the gating window on the TTL channel
        self.dds_excitation.sw.off()

        delay(1*ms)

        # turn AOMs back on
        self.dds_AOM_A1.sw.on()
        self.dds_AOM_A2.sw.on()
        self.dds_AOM_A3.sw.on()
        self.dds_AOM_A4.sw.on()
        self.dds_AOM_A5.sw.on()
        self.dds_AOM_A6.sw.on()

        # todo: delete
        # self.core.wait_until_mu(loop_over_mu+1000)
        # at_mu(loop_over_mu+1000)
        # self.print_async("out of the loop",now_mu() - loop_start_mu)

        delay(1*ms)

        rtio_log("2nd_shot_block",1)
        # self.print_async("second readout",now_mu() - loop_start_mu) # todo: delete
        with sequential:

            self.ttl_SPCM_gate.off() # enables the SPCM
            self.ttl_repump_switch.off()  # turns the RP AOM on

            # take the second shot
            self.zotino0.set_dac(
                [self.AZ_bottom_volts_RO, self.AZ_top_volts_RO, self.AX_volts_RO, self.AY_volts_RO],
                channels=self.coil_channels)
            delay(0.1 * ms)


            second_shot(self)
        rtio_log("2nd_shot_block",0)

        end_measurement(self)
        for val in excitation_counts_array:
            self.append_to_dataset('excitation_counts', val)
        for val in excitation_counts_array1:
            self.append_to_dataset('excitation_counts1', val)

        delay(10*ms)

    self.dds_FORT.sw.off()

###############################################################################
# 3. DIAGNOSTIC FUNCTIONS
# These are functions that are used for various tests, but are not typical
# experiments. You might not want to run them with GeneralVariableScan, but you
# can import these functions from other files. The advantage of having them in
# experiment_functions is that they can benefit from the other functions defined
# here.
###############################################################################

@kernel
def FORT_monitoring_with_Luca_experiment(self):
    """
    A modified version of atom_loading_experiment for monitoring the FORT scattering.
    This is most easily run with tests/MonitorFORTPowerWithLuca.py

    Load a MOT, load a single atom, readout, wait self.t_delay_between_shots, readout again.

    For this experiment, we were still using the MM fiber monitor (i.e. after the polarizer) to
    feed back to the FORT power. We used this experiment to characterize how well the signal
    from an APD monitoring the scattered 852 nm light in the chamber correlated with the signal
    seen by the Luca in an ROI restricted to a section of the parabolic mirror tube.

    The assumed connection of the Sampler card and channel used to monitor the APD is given
    in load_MOT_and_FORT_for_Luca_scattering_measurement.

    ----
    Alternatively, if you are using the APD to feedback to the FORT:
    The MM fiber is monitored with the channel specified in load_MOT_and_FORT_for_Luca_scattering_measurement.

    The purpose of this experiment is to be able to compare the normalized FORT power recorded
    by the dds_FORT feedback channel, the scattering seen by the camera, and the voltage of
    the detector monitoring the MM fiber after the polarizer. Rotating the 852 nm
    motorized waveplates, e.g. with the APT program, we can check whether the scattering in the
    chamber seen with the Luca correlates with polarization. This is a way to vet our APD
    feedback scheme.

    ----
    For analysis, see FORT feedback/monitor_FORT_scattering_and_Raman_light.ipynb

    :param self: an experiment instance.
    :return:
    """

    self.core.reset()

    self.set_dataset("photocounts_FORT_loading", [0.0], broadcast=True)
    self.set_dataset("photocounts_FORT_and_MOT", [0.0], broadcast=True)
    self.set_dataset("photocounts_FORT_science", [0.0], broadcast=True)
    self.set_dataset("APD_FORT_volts_loading", [0.0], broadcast=True)
    self.set_dataset("APD_FORT_volts_science", [0.0], broadcast=True)

    self.counts = 0
    self.counts2 = 0

    self.require_D1_lock_to_advance = False # override experiment variable
    self.require_atom_loading_to_advance = False # override experiment variable

    self.set_dataset(self.count_rate_dataset,
                     [0.0],
                     broadcast=True)

    delay(100*ms)
    now = now_mu()
    for i in range(self.warm_up_shots):
        at_mu(now+i*self.core.seconds_to_mu(500*ms))
        self.ttl_Luca_trigger.pulse(5 * ms)
        # now = now_mu()

    self.core.wait_until_mu(now_mu())

    self.measurement = 0
    while self.measurement < self.n_measurements:

        if self.enable_laser_feedback:
            self.laser_stabilizer.run(monitor_only=self.no_feedback)  # this tunes the MOT and FORT AOMs

        load_MOT_and_FORT_for_Luca_scattering_measurement(self)

        delay(0.1*ms)
        self.zotino0.set_dac(
            [self.AZ_bottom_volts_RO, self.AZ_top_volts_RO, self.AX_volts_RO, self.AY_volts_RO],
            channels=self.coil_channels)

        # set the FORT AOM to the science setting. this is only valid if we have run
        # feedback to reach the corresponding setpoint first, which in this case, happened in load_MOT_and_FORT
        if not self.no_feedback:
            self.dds_FORT.set(frequency=self.f_FORT,
                                    amplitude=self.stabilizer_FORT.amplitudes[1])

        # set the cooling DP AOM to the readout settings
        self.dds_cooling_DP.set(frequency=self.f_cooling_DP_RO,
                                amplitude=self.ampl_cooling_DP_MOT*self.p_cooling_DP_RO)

        # take the first shot
        if not self.MOT_light_off:
            self.dds_cooling_DP.sw.on()
        with parallel:
            self.ttl_Luca_trigger.pulse(5 * ms)
            t_gate_end = self.ttl_SPCM0.gate_rising(self.t_SPCM_first_shot)
        self.counts = self.ttl_SPCM0.count(t_gate_end)
        delay(1 * ms)
        self.dds_cooling_DP.sw.off()

        delay(self.t_delay_between_shots)

        # take the second shot
        if not self.MOT_light_off:
            self.dds_cooling_DP.sw.on()
        # with parallel:
            # self.ttl_Luca_trigger.pulse(5 * ms)
        t_gate_end = self.ttl_SPCM0.gate_rising(self.t_SPCM_second_shot)
        self.counts2 = self.ttl_SPCM0.count(t_gate_end)

        delay(1 * ms)

        end_measurement(self)

        # update experiment-specific datasets:
        self.append_to_dataset("photocounts_FORT_loading", self.counts_FORT_loading)
        self.append_to_dataset("photocounts_FORT_and_MOT", self.counts_FORT_and_MOT)
        self.append_to_dataset("photocounts_FORT_science", self.counts_FORT_science)
        self.append_to_dataset("APD_FORT_volts_loading", self.APD_FORT_volts_loading)
        self.append_to_dataset("APD_FORT_volts_science", self.APD_FORT_volts_science)

    self.dds_FORT.sw.off()

@kernel
def atom_loading_and_waveplate_rotation_experiment(self):
    """
    The most basic two-readout single atom experiment.

    Load a MOT, load a single atom, readout, wait self.t_delay_between_shots, readout again.

    :param self: an experiment instance.
    :return:
    """

    self.core.reset()

    self.counts = 0
    self.counts2 = 0

    self.require_D1_lock_to_advance = False # override experiment variable
    self.require_atom_loading_to_advance = False # override experiment variable

    self.set_dataset(self.count_rate_dataset,
                     [0.0],
                     broadcast=True)

    self.measurement = 0
    while self.measurement < self.n_measurements:

        self.FORT_HWP.move_by(self.hwp_degrees_to_move_by) # degrees
        self.FORT_QWP.move_by(self.qwp_degrees_to_move_by) # degrees
        delay(10*ms)

        if self.enable_laser_feedback:
            self.laser_stabilizer.run()  # this tunes the MOT and FORT AOMs

        load_MOT_and_FORT(self)

        delay(0.1*ms)
        self.zotino0.set_dac(
            [self.AZ_bottom_volts_RO, self.AZ_top_volts_RO, self.AX_volts_RO, self.AY_volts_RO],
            channels=self.coil_channels)

        # set the FORT AOM to the science setting. this is only valid if we have run
        # feedback to reach the corresponding setpoint first, which in this case, happened in load_MOT_and_FORT

        self.dds_FORT.set(frequency=self.f_FORT,
                                amplitude=self.stabilizer_FORT.amplitudes[1])

        # set the cooling DP AOM to the readout settings
        self.dds_cooling_DP.set(frequency=self.f_cooling_DP_RO,
                                amplitude=self.ampl_cooling_DP_MOT*self.p_cooling_DP_RO)

        if not self.no_first_shot:
            # take the first shot
            self.dds_cooling_DP.sw.on()
            t_gate_end = self.ttl_SPCM0.gate_rising(self.t_SPCM_first_shot)
            self.counts = self.ttl_SPCM0.count(t_gate_end)
            delay(1 * ms)
            self.dds_cooling_DP.sw.off()

        if self.t_FORT_drop > 0:
            self.dds_FORT.sw.off()
            delay(self.t_FORT_drop)
            self.dds_FORT.sw.on()

        delay(self.t_delay_between_shots)

        # take the second shot
        self.dds_cooling_DP.sw.on()
        t_gate_end = self.ttl_SPCM0.gate_rising(self.t_SPCM_second_shot)
        self.counts2 = self.ttl_SPCM0.count(t_gate_end)

        delay(1 * ms)

        end_measurement(self)

    self.dds_FORT.sw.off()