import os
import subprocess
import numpy as np

def run(filepath, script):
    '''Creates and executes a file given the script at the given filepath
    '''
    with open(filepath, "w") as f:
        f.write("#!/bin/bash\n")    # Add the shebang line
        f.write(script)
    os.chmod(filepath, 0o755)       # Make the script executable

    subprocess.call(f'./{filepath}')

def calc_air_mass(zenith):
    '''Returns the airmass given the zenith angle in radians
    '''
    return 1/np.cos(zenith)


def calc_zenith(air_mass):
    '''Returns the zenith angle in radians given the airmass 
    '''
    return np.arccos(1/air_mass)

def DD_CubicHermiteSpline(eval_airmass, eval_nscale, data_dict, reverse=False):
    '''Returns the interpolation of the data given an airmass and nscale value

    `reverse` - set True to reverse the order of operation
    '''

    from scipy.interpolate import CubicHermiteSpline
    from scipy.interpolate import RegularGridInterpolator

    Nscale_map = data_dict['Nscale']['map'][::2]
    Tb_scalar_field = data_dict['Tb_scalar_field'][::2,::2]
    Nscale_jacobian = data_dict['Nscale']['jacobian'][::2,::2]
    airmass_map = data_dict['airmass']['map'][::2]
    freq_map = data_dict['freq']['map']
    airmass_jacobian = data_dict['airmass']['jacobian'][::2,::2]

    init_interp_func = CubicHermiteSpline(
        x=Nscale_map if reverse else airmass_map,
        y=Tb_scalar_field,
        dydx=Nscale_jacobian if reverse else airmass_jacobian,
        axis=0 if reverse else 1,
    )

    first_eval = init_interp_func(eval_nscale if reverse else eval_airmass)

    # Interpolate for nscale Jacobian at the chosen airmass
    jacob_interp_func = RegularGridInterpolator(
        points=(Nscale_map, airmass_map, freq_map),
        values=airmass_jacobian if reverse else Nscale_jacobian,
        method="linear",
    )

    x,y,z = np.meshgrid(
        eval_nscale if reverse else Nscale_map,
        airmass_map if reverse else eval_airmass,
        freq_map,
        indexing='ij',
    )

    mod_jacobian = jacob_interp_func(
        (x.flatten(),y.flatten(),z.flatten())
    ).reshape(x.shape)

    final_interp_func = CubicHermiteSpline(
        x=airmass_map if reverse else Nscale_map,
        y=first_eval,
        dydx=mod_jacobian,
        axis=1 if reverse else 0,
    )

    return final_interp_func(eval_airmass if reverse else eval_nscale)
    
def dictionarify(Nscale_points, Nscale_map, Nscale_jacobian, 
                airmass_points, airmass_map, airmass_jacobian, 
                freq_points, freq_map, Tb_scalar_field):
    '''Returns all the provided data as a dictionary in the following format:
            ```
            {'airmass':{
                'map':airmass_map,
                'jacobian':airmass_jacobian,
                'points':airmass_points
            },
            'Nscale':{
                'map':Nscale_map,
                'jacobian':Nscale_jacobian,
                'points':Nscale_points
            },
            'freq':{
                'map':freq_map,
                'points':freq_points
            },
            'Tb_scalar_field':Tb_scalar_field
            }
            ```
    '''
    
    return {'airmass':{
                'map':airmass_map,
                'jacobian':airmass_jacobian,
                'points':airmass_points
            },
            'Nscale':{
                'map':Nscale_map,
                'jacobian':Nscale_jacobian,
                'points':Nscale_points
            },
            'freq':{
                'map':freq_map,
                'points':freq_points
            },
            'Tb_scalar_field':Tb_scalar_field
            }