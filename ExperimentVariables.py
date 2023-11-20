from artiq.experiment import *
import numpy as np
from collections import namedtuple


def setattr_variables(experiment):
    """
    Set variables in your experiment from existing datasets of the same names.

    Usage: in the build method of your experiment, declare a list of variables
    then call this method (having imported it at the top) with experiment=self.

    :param experiment: an ARTIQ Experiment object with an attribute variables which is a list
        of strings which are the variables names we want. It is assumed that the variables
        have already been defined in ExperimentVariables.
    :return:
    """
    for var in experiment.variables:
        try:  # get the value of the variable from the dataset, if it exists
            value = experiment.get_dataset(var)
            setattr(experiment, var, value)
        except Exception as e:
            if type(e) == KeyError:  # if the variable does not exist
                print(f"Couldn't find variable {e}! Did you define it in vars_list in ExperimentVariables?")
            else:
                print(f"Exception {e}") # todo: replace with raise statement


class ExperimentVariables(EnvExperiment):

    def build(self):

        Variable = namedtuple("Variable", "name value value_type kwargs group")
        """
        An object for defining an experiment variable. This simplifies creating the datasets for each var.
        
        name: the variable name. must be a valid python name
        value: the value of the variable. this is only used the very first time the code is run after you add
            a new variable. Every subsequent run, the value will be pulled from the dataset that is created.
        value_type: the artiq value function, e.g. NumberValue or StringValue
        kwargs: a dictionary of keyword arguments used by the value function.
        group: set to None or a string. If a string, all Variables with this same string will be grouped 
            in the GUI.
            
        Example:
        Variable("f_FORT", 210.0 * MHz, NumberValue, {'type': 'float', 'unit': 'MHz'}, "FORT switching AOM")
        """

        # the list of variables which we will use to make datasets of the same name stored in an hdf file.
        # these datasets can be referenced by other experiments instead of declaring the variable locally in each
        # experiment.
        self.vars_list = [
            # FORT AOM
            Variable("f_FORT", 210.0 * MHz, NumberValue, {'type': 'float', 'unit': 'MHz'}, "FORT AOM"),
            Variable("p_FORT_loading", 3, NumberValue, {'type': 'float', 'unit': "dBm", 'scale': 1, 'ndecimals': 1},
                                  "FORT AOM"),
            Variable("p_FORT_RO", 3, NumberValue, {'type': 'float', 'unit': "dBm", 'scale': 1, 'ndecimals': 1},
                     "FORT AOM"),
            Variable("p_FORT_PGC", 3, NumberValue, {'type': 'float', 'unit': "dBm", 'scale': 1, 'ndecimals': 1},
                     "FORT AOM"),

            # Cooling double pass AOM
            Variable("f_cooling_DP_MOT", 111.0 * MHz, NumberValue, {'type': 'float', 'unit': 'MHz'},
                     "Cooling double pass AOM"),
            Variable("f_cooling_DP_RO", 111.0 * MHz, NumberValue, {'type': 'float', 'unit': 'MHz'},
                     "Cooling double pass AOM"),
            Variable("f_cooling_DP_PGC", 111.0 * MHz, NumberValue, {'type': 'float', 'unit': 'MHz'},
                     "Cooling double pass AOM"),
            Variable("p_cooling_DP_MOT", -4, NumberValue, {'type': 'float', 'unit': "dBm", 'scale': 1, 'ndecimals': 1},
                     "Cooling double pass AOM"),
            Variable("p_cooling_DP_RO", -4, NumberValue, {'type': 'float', 'unit': "dBm", 'scale': 1, 'ndecimals': 1},
                     "Cooling double pass AOM"),
            Variable("p_cooling_DP_PGC", -4, NumberValue, {'type': 'float', 'unit': "dBm", 'scale': 1, 'ndecimals': 1},
                     "Cooling double pass AOM"),

            # Cooling single pass AOM
            Variable("f_D1_pumping_SP", 130.0 * MHz, NumberValue, {'type': 'float', 'unit': 'MHz'},
                     "Cooling single pass AOM"),
            Variable("p_D1_pumping_SP", 1, NumberValue, {'type': 'float', 'unit': "dBm", 'scale': 1, 'ndecimals': 1},
                     "Cooling single pass AOM"),

            # D1 optical pumping, pumping repumper, and excitation
            Variable("f_D1_pumping_SP", 100.00 * MHz, NumberValue, {'type': 'float', 'unit': 'MHz'},
                     "OP and excitation AOMs"),
            Variable("p_D1_pumping_SP", -8.0, NumberValue, {'type': 'float', 'unit': "dBm", 'scale': 1, 'ndecimals': 1},
                     "OP and excitation AOMs"),
            Variable("f_pumping_repump", 345.10 * MHz, NumberValue, {'type': 'float', 'unit': 'MHz'},
                     "OP and excitation AOMs"),
            Variable("p_pumping_repump", -8.0, NumberValue, {'type': 'float', 'unit': "dBm", 'scale': 1, 'ndecimals': 1},
                     "OP and excitation AOMs"),
            Variable("f_excitation", 150.50 * MHz, NumberValue, {'type': 'float', 'unit': 'MHz'},
                     "OP and excitation AOMs"),
            Variable("p_excitation", -8.0, NumberValue, {'type': 'float', 'unit': "dBm", 'scale': 1, 'ndecimals': 1},
                     "OP and excitation AOMs"),

            # D1 optical pumping
            Variable("f_D1_pumping_SP", 150.5 * MHz, NumberValue, {'type': 'float', 'unit': 'MHz'},
                     "Repump AOMs"),
            Variable("p_D1_pumping_SP", 3, NumberValue, {'type': 'float', 'unit': "dBm", 'scale': 1, 'ndecimals': 1},
                     "Repump AOMs"),

            # Fiber AOMs
            Variable("AOM_A1_freq", 78.5 * MHz, NumberValue, {'type': 'float', 'unit': 'MHz', 'ndecimals': 3},
                     "Fiber AOMs"),
            Variable("AOM_A1_power", 0, NumberValue, {'type': 'float', 'unit': "dBm", 'scale': 1, 'ndecimals': 1},
                     "Fiber AOMs"),
            Variable("AOM_A2_freq", 78.0 * MHz, NumberValue, {'type': 'float', 'unit': 'MHz', 'ndecimals': 3},
                     "Fiber AOMs"),
            Variable("AOM_A2_power", 0, NumberValue, {'type': 'float', 'unit': "dBm", 'scale': 1, 'ndecimals': 1},
                     "Fiber AOMs"),
            Variable("AOM_A3_freq", 78.503 * MHz, NumberValue, {'type': 'float', 'unit': 'MHz', 'ndecimals': 3},
                     "Fiber AOMs"),
            Variable("AOM_A3_power", -3, NumberValue, {'type': 'float', 'unit': "dBm", 'scale': 1, 'ndecimals': 1},
                     "Fiber AOMs"),
            Variable("AOM_A4_freq", 78.504 * MHz, NumberValue, {'type': 'float', 'unit': 'MHz', 'ndecimals': 3},
                     "Fiber AOMs"),
            Variable("AOM_A4_power", 0, NumberValue, {'type': 'float', 'unit': "dBm", 'scale': 1, 'ndecimals': 1},
                     "Fiber AOMs"),
            Variable("AOM_A5_freq", 78.505 * MHz, NumberValue, {'type': 'float', 'unit': 'MHz', 'ndecimals': 3},
                     "Fiber AOMs"),
            Variable("AOM_A5_power", 0, NumberValue, {'type': 'float', 'unit': "dBm", 'scale': 1, 'ndecimals': 1},
                     "Fiber AOMs"),
            Variable("AOM_A6_freq", 78.506 * MHz, NumberValue, {'type': 'float', 'unit': 'MHz', 'ndecimals': 3},
                     "Fiber AOMs"),
            Variable("AOM_A6_power", 0, NumberValue, {'type': 'float', 'unit': "dBm", 'scale': 1, 'ndecimals': 1},
                     "Fiber AOMs"),

            # Coils - MOT
            Variable("AZ_bottom_volts_MOT", 1.02, NumberValue, {'type': 'float', 'unit': 'V', 'ndecimals':3}, "MOT coil settings"),
            Variable("AZ_top_volts_MOT", -3.4, NumberValue, {'type': 'float', 'unit': 'V', 'ndecimals':3}, "MOT coil settings"),
            Variable("AX_volts_MOT", -0.11, NumberValue, {'type': 'float', 'unit': 'V', 'ndecimals':3}, "MOT coil settings"),
            Variable("AY_volts_MOT", 0, NumberValue, {'type': 'float', 'unit': 'V', 'ndecimals':3}, "MOT coil settings"),

            # Coils - PGC
            Variable("AZ_bottom_volts_PGC", 0, NumberValue, {'type': 'float', 'unit': 'V', 'ndecimals':3}, "MOT coil settings"),
            Variable("AZ_top_volts_PGC", 0, NumberValue, {'type': 'float', 'unit': 'V', 'ndecimals':3}, "MOT coil settings"),
            Variable("AX_volts_PGC", -0.11, NumberValue, {'type': 'float', 'unit': 'V', 'ndecimals':3}, "MOT coil settings"),
            Variable("AY_volts_PGC", 0, NumberValue, {'type': 'float', 'unit': 'V', 'ndecimals':3}, "MOT coil settings"),

            # Coils - Readout
            Variable("AZ_bottom_volts_RO", 0, NumberValue, {'type': 'float', 'unit': 'V', 'ndecimals':3}, "MOT coil settings"),
            Variable("AZ_top_volts_RO", 0, NumberValue, {'type': 'float', 'unit': 'V', 'ndecimals':3}, "MOT coil settings"),
            Variable("AX_volts_RO", -0.11, NumberValue, {'type': 'float', 'unit': 'V', 'ndecimals':3}, "MOT coil settings"),
            Variable("AY_volts_RO", 0, NumberValue, {'type': 'float', 'unit': 'V', 'ndecimals':3}, "MOT coil settings"),

            # Timing
            Variable("t_MOT_loading", 500*ms, NumberValue, {'type': 'float', 'unit': 'ms'}, "Timing"),
            Variable("t_FORT_loading", 50*ms, NumberValue, {'type': 'float', 'unit': 'ms'}, "Timing"),
            Variable("t_SPCM_exposure", 50 * ms, NumberValue, {'type': 'float', 'unit': 'ms'}, "Timing"),
            Variable("t_SPCM_first_shot", 10 * ms, NumberValue, {'type': 'float', 'unit': 'ms'}, "Timing"),
            Variable("t_SPCM_second_shot", 10 * ms, NumberValue, {'type': 'float', 'unit': 'ms'}, "Timing"),
            Variable("t_PGC_in_MOT", 50 * ms, NumberValue, {'type': 'float', 'unit': 'ms'}, "Timing"),

            # Set points
            Variable("set_point_PD0_AOM_cooling_DP", 0.784, NumberValue, {'type': 'float','ndecimals':3}, "Set points"),
            Variable("set_point_fW_AOM_A1", 0.768, NumberValue, {'type':'float','ndecimals':3}, "Set points"),
            Variable("set_point_fW_AOM_A2", 0.644, NumberValue, {'type': 'float','ndecimals':3}, "Set points"),
            Variable("set_point_fW_AOM_A3", 0.956, NumberValue, {'type': 'float','ndecimals':3}, "Set points"),
            Variable("set_point_fW_AOM_A4", 0.588, NumberValue, {'type': 'float','ndecimals':3}, "Set points"),
            Variable("set_point_PD5_AOM_A5", 0.214, NumberValue, {'type': 'float','ndecimals':3}, "Set points"),
            Variable("set_point_PD6_AOM_A6", 0.296, NumberValue, {'type': 'float','ndecimals':3}, "Set points"),
            Variable('set_point_FORT_MM', 0.272, NumberValue, {'type': 'float','ndecimals':3}, "Set points"),

            # Plotting
            Variable("MOT_beam_monitor_points", 100, NumberValue, {'type': 'int', 'ndecimals': 0, 'scale': 1, 'step':1},
                     "Plotting"),
            # Plotting
            Variable("ignore_first_n_histogram_points", 10, NumberValue, {'type': 'int', 'ndecimals': 0, 'scale': 1,
                     'step': 1}, "Plotting"),

            # Laser feedback
            Variable("Luca_trigger_for_feedback_verification", False, BooleanValue, {}, "Laser feedback"),
            Variable("enable_laser_feedback", False, BooleanValue, {}, "Laser feedback"),
            Variable("feedback_dds_list", "['dds_AOM_A1', 'dds_AOM_A2', 'dds_AOM_A3', "
                                          "'dds_AOM_A4','dds_AOM_A4','dds_AOM_A5',"
                                      "'dds_AOM_A6','dds_cooling_DP']", StringValue, {}, "Laser feedback"),
            Variable("aom_feedback_averages", 4, NumberValue, {'type': 'int', 'ndecimals': 0, 'scale': 1, 'step':1},
                     "Laser feedback"),
            Variable("aom_feedback_iterations", 4, NumberValue, {'type': 'int', 'ndecimals': 0, 'scale': 1, 'step':1},
                     "Laser feedback")

            # File saving # todo: move away from saving csv files and save instead to the experiment's hdf
        ]

        # can only call get_dataset in build, but can only call set_dataset in run. so
        # we need to see which datasets exist here, create new ones if there are new vars,
        # and then in run we will set all the datasets with the values from the GUI.
        for var in self.vars_list:
            try:  # get the value of the variable from the dataset, if it exists
                value = self.get_dataset(var.name)
                self.setattr_argument(var.name, var.value_type(value, **var.kwargs), var.group)
            except Exception as e:
                if type(e) == KeyError:  # if the variable does not exist, create the dataset
                    print(f"Found new variable {e}! Adding argument.")
                    self.setattr_argument(var.name, var.value_type(var.value, **var.kwargs), var.group)
                else:
                    print(f"Exception {e}")

    def run(self):

        for var in self.vars_list:
            self.set_dataset(var.name, getattr(self, var.name), broadcast=True, persist=True)