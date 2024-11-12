"""
Control two K10CR1 units with only one NDSP server, where the calls to the
NDSP object are made from a kernel.

1. connect thorlabs K10CR1 rotators to your machine with USB, and make sure
the serial numbers in the device_db match the rotators you connected.

2. open a terminal in the same directory as this script, 
enter your artiq virtual environment, then run the 
NDSP server for this test with

    python launcher_multi_rotor.py

3. In another terminal in this directory in the artiq environment,
run this script with artiq:

    artiq_run example_multi_k10cr1.py --device-db=.\device_db.py

If you wanted to connect to multiple K10CR1 units, at least if we stick to the
way this example is structured using the K10CR1_NDSP_Driver class, you would need 
to launch a separate server for every Thorlabs rotator, which is probably not a 
good solution. A better solution would have only one server for all the K10CR1 units,
and this is what the K10CR1_Multi_NDSP_Driver tries to accomplish. See the multi
examples."""

# here's how you do ndsp stuff in your artiq scripts
from artiq.experiment import *
from time import sleep


class K10CR1Example(EnvExperiment):

    def build(self):
        self.setattr_device("core")
        self.setattr_device("led0")

        try:
            self.setattr_device("k10cr1_ndsp")
        except Exception as e:
            print(f"Error connecting to device {e}")
    
    # if you want to call NDSP functions from the kernel and return things,
    # you need to define wrapper functions. if you call the NDSP functions
    # directly, you'll get an error saying that the expression of NoneType
    # can't be unified with float (or whatever your return type is). 
    def get_rotator_position(self, name: TStr) -> TFloat:
        """wrapper function"""
        positions = self.k10cr1_ndsp.get_position(name)
        return positions

    @kernel
    def experiment_function(self):

        self.core.reset()

        # get position and move the 780 QWP
        print('780_QWP at', self.get_rotator_position('780_QWP'))
        print('780_HWP at', self.get_rotator_position('780_HWP'))
        delay(100*ms)

        self.k10cr1_ndsp.move_by(20, '780_QWP')
        self.k10cr1_ndsp.move_by(15, '780_HWP')

        delay(2*s)  # todo: wait in a more sophisticated way 

        # do some stuff with the Sinara hardware
        # for measurement in range(self.n_measurements):
        #     your tomography experiment goes here...
        #

        for i in range(100):
            self.led0.pulse(0.1*s)
            delay(0.1*s)

    def initialize_hardware(self):
        print("setting up hardware!")

    def initialize_datasets(self):
        print("setting up datasets!")

    # run is not on the kernel but it calls methods that are
    def run(self):
        """
        hint hint: does this remind you of GeneralVariableScan?
        """

        self.initialize_datasets()
        self.initialize_hardware()

        # our experiment on the kernel
        self.experiment_function()

