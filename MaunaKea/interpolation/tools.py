import os
import subprocess
import numpy as np

def run(filepath, script):
    '''Creates an execuated file given the script and executes it at the given filepath
    '''
    with open(filepath, "w") as f:
        f.write("#!/bin/bash\n")    # Add the shebang line
        f.write(script)
    os.chmod(filepath, 0o755)       # Make the script executable

    subprocess.call(f'./{filepath}')

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

    if reverse:
        # Interpolate for chosen nscale
        interp_func = CubicHermiteSpline(
            x=Nscale_map, y=Tb_scalar_field, dydx=Nscale_jacobian, axis=0
        )

        Nscale_eval_grid = interp_func(eval_nscale)
        
        # Interpolate for airmass Jacobian at the chosen nscale value
        interp_func = RegularGridInterpolator(
            points=(Nscale_map, airmass_map, freq_map), values=airmass_jacobian, method="linear"
        )
        
        x,y,z = np.meshgrid(eval_nscale, airmass_map, freq_map, indexing='ij')
        airmass_jacobian_fixed_nscale = interp_func(
            (x.flatten(),y.flatten(),z.flatten())
        ).reshape(Nscale_eval_grid.shape)

        # Interpolate for 2D spectrum using chosen airmass and fixed airmass Jacobian
        interp_func = CubicHermiteSpline(
            x=airmass_map, y=Nscale_eval_grid, dydx=airmass_jacobian_fixed_nscale, axis=0
        )
        interp_val = interp_func(eval_airmass).reshape(freq_map.shape)
        
        return interp_val
    else:
        # Interpolate for chosen airmass
        interp_func = CubicHermiteSpline(
            x=airmass_map, y=Tb_scalar_field, dydx=airmass_jacobian, axis=1
        )

        airmass_eval_grid = interp_func(eval_airmass)
        
        # Interpolate for nscale Jacobian at the chosen airmass
        interp_func = RegularGridInterpolator(
            points=(Nscale_map, airmass_map, freq_map), values=Nscale_jacobian, method="linear"
        )
        
        x,y,z = np.meshgrid(Nscale_map, eval_airmass, freq_map, indexing='ij')
        nscale_jacobian_fixed_airmass = interp_func(
            (x.flatten(),y.flatten(),z.flatten())
        ).reshape(airmass_eval_grid.shape)

        # Interpolate for 2D spectrum using chosen nscale and fixed nscale Jacobian
        interp_func = CubicHermiteSpline(
            x=Nscale_map, y=airmass_eval_grid, dydx=nscale_jacobian_fixed_airmass, axis=0
        )
        interp_val = interp_func(eval_nscale).reshape(freq_map.shape)
        
        return interp_val
    
    
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