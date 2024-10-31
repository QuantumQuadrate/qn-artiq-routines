"""
open your artiq environment in the same directory as this
script, and run

    artiq_run ndsp_example.py --device-db=.\device_db.py
"""

# here's how you do ndsp stuff in your artiq scripts
from artiq.experiment import *

class NDSPExample(EnvExperiment):

    def build(self):
        self.setattr_device("core")

        try:
            self.setattr_device("example_ndsp")
        except Exception as e:
            print(f"Error connecting to device {e}")
    
    def run(self):
        self.example_ndsp.do_something()