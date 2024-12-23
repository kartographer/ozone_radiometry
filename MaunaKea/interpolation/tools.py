import os
import subprocess
import numpy as np

def generate_Tb_spectrum(filename, script, WINDOWS=False):

    if WINDOWS:
        filename = filename[:-2] + 'bat'
        with open(filename, "w") as f:
            f.write(script)

        os.startfile(filename)

    else:
        with open(filename, "w") as f:
            f.write("#!/bin/bash\n")    # Add the shebang line
            f.write(script)
        os.chmod(filename, 0o755)       # Make the script executable

        subprocess.call(f'./{filename}')

def calc_air_mass(zenith):
    '''Returns the airmass given the zenith angle in degrees
    '''
    air_mass = 1/np.cos(np.radians(zenith))
    return air_mass

def calc_zenith(air_mass):
    '''Returns the zenith angle in degrees given the airmass 
    '''
    zenith = np.degrees(np.arccos(1/air_mass))
    return zenith