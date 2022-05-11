
import yt
import math
from yt import YTArray

from yt.utilities.cosmology import Cosmology

co = Cosmology(hubble_constant=0.7, omega_matter=0.3, 
               omega_lambda=0.7, omega_curvature=0.0)

import numpy as np 
from yt.units import G 
import matplotlib.pyplot as plt
import os
import array


import matplotlib
from scipy.interpolate import RectBivariateSpline
import pandas as pd
import matplotlib.colors as mcolors
import os
from scipy import stats
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
from os.path import expanduser
home = expanduser("~")
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec

#from skspatial.plotting import plot_3d
#from skspatial.objects import Point, Vector, Plane


from matplotlib import rcParams
from yt.utilities.cosmology import Cosmology




def young_star_filter(pfilter,data):
    age = data.ds.current_time - data[pfilter.filtered_type, "particle_creation_time"]
    filter = age.in_units('Gyr') < 5
    return filter

def calculate_center(ds):
    ad=ds.all_data()
    c1=ad.quantities.max_location(('deposit', 'stars_density'))
    print(c1)
    c2=[c1[1],c1[2],c1[3]]
    print(c2)
    spe0=ds.sphere(c2,(200,'kpc'))
    c3=spe0.quantities.center_of_mass(use_gas=False,use_particles=True, particle_type = "all")

    spe1=ds.sphere(c3,(35,'kpc'))
    c4=spe1.quantities.center_of_mass(use_gas=False,use_particles=True, particle_type = "stars")
    print(c4)

    spe3= ds.sphere(c4, (15, 'kpc'))
    c6=spe3.quantities.center_of_mass(use_gas=False,use_particles=True, particle_type = "stars")
    print(c6.in_units("cm"))

    spe4=ds.sphere(c6,(3,'kpc'))
    c7=spe4.quantities.center_of_mass(use_gas=False,use_particles=True, particle_type = "stars")
    print(c7.in_units("cm"))

    spe5=ds.sphere(c7,(0.2,'kpc'))
    center=spe5.quantities.center_of_mass(use_gas=False,use_particles=True, particle_type = "stars")
    print(center.in_units("cm"))
    return center.in_units("cm")


def calculate_angular_momentum(ds, center, Rvir):
    sp5 = ds.sphere(center, (0.1*Rvir))
    am = sp5.quantities.angular_momentum_vector(use_gas=False,use_particles=True,particle_type= 'young_stars' )
    angular_momentum = -am
    return angular_momentum

def calculate_rvir(ds,center, delta_vir):
    sp=ds.sphere(center,(250, "kpc"))
    GasMassType_to_use = "cell_mass"

    profile_part= yt.create_profile(sp, ('all',"particle_position_spherical_radius"),
    ('all','particle_mass'), n_bins=120, weight_field = None,extrema={('all',"particle_position_spherical_radius"): (2e20, 4e24)})#(1e18, 5.0e23)})
    profile_part_cumsum=np.cumsum(profile_part["all","particle_mass"])
    binnum=len(profile_part.x)
    profile_gas = yt.create_profile(sp, "radius", GasMassType_to_use, weight_field =None,
                                    n_bins=binnum,extrema={'radius': (2e20, 4e24)})
    profile_gas_cumsum=np.cumsum(profile_gas[GasMassType_to_use])
    dens_bin=[0.0]*len(profile_part.x)
    dens_bin[0]=(profile_part["all","particle_mass"][0]+profile_gas[GasMassType_to_use][0])/(4./3.*3.14159265*np.power(profile_part.x[0],3))
    dens_bin[0]=(profile_part_cumsum[0]+profile_gas_cumsum[0])/(4./3.*3.14159265*np.power(profile_part.x[0],3))
    #print(np.max(sp[("gas","radius")]),np.min(sp[("gas","radius")]))
    for i in range(1,len(dens_bin)):
        dens_bin[i]=(profile_part_cumsum[i]+profile_gas_cumsum[i])/(4./3.*3.14159265*(np.power(profile_part.x[i],3)+np.power(profile_part.x[i-1],3))/2.)
        #delta_vir=333.#(18.*3.14159265**2.+(co.omega_matter(pf.current_redshift)-1)*82-39.*(co.omega_matter(pf.current_redshift)-1)**2.)
        if i == len(dens_bin)-1:
            print('Rvir Not found, so Rvir=250kpc')
            rvir=YTArray(10.,'kpc')
            sp1=ds.sphere(center,rvir)
        if 0. < dens_bin[i] <= delta_vir*co.critical_density(ds.current_redshift):
            print(dens_bin[i], delta_vir*co.critical_density(ds.current_redshift),dens_bin[i-1])
            rvir=profile_part.x[i]
            print('Rvir=',rvir.in_units('kpc'))
    #        sp1=ds.sphere(center,profile_part.x[i])
    #        baryon_mass, particle_mass = sp.quantities.total_quantity([GasMassType_to_use, "particle_mass"])
    #        print("Total mass in sphere is %0.3e Msun (gas = %0.3e Msun, particles = %0.3e Msun)" %  ((baryon_mass + particle_mass).in_units('Msun'), baryon_mass.in_units('Msun'), particle_mass.in_units('Msun')))
        #       Mv=(baryon_mass + particle_mass).in_units('Msun')
            break
    print("Rvir: ", rvir.in_units("kpc"))
    return rvir.in_units("kpc")


def calculate_mean_velocity(data):
        mean_vx = data[('stars', 'particle_velocity_x')].mean()
        mean_vy = data[('stars', 'particle_velocity_y')].mean()
        mean_vz = data[('stars', 'particle_velocity_z')].mean()
        return mean_vx.in_units("km/s"), mean_vy.in_units("km/s"), mean_vz.in_units("km/s")



def metallicity_star_alfaH(field, data):
    return (np.log10(data[('stars', 'particle_metallicity1')]/0.0161))
                
def metallicity_star_FeH(field, data):
    return (np.log10(data[('stars', 'particle_metallicity2')]/0.00178))

def metallicity_star_AlfaFe(field, data):
    return (np.log10(data[('stars', 'particle_metallicity1')]/data[('stars', 'particle_metallicity2')])-np.log10(0.0161/0.00178))


def data_to_csv_pre (size, grupo):
    print(grupo)
    ID = np.array(size[grupo, "particle_index"])
    X = np.array(size[grupo,"position_x_change_coord"].in_units("kpc"))
    VX = np.array(size[grupo, "velocity_x"].in_units("km/s"))
    Y = np.array(size[grupo, "position_y_change_coord"].in_units("kpc"))
    VY = np.array(size[grupo, "velocity_y"].in_units("km/s"))
    Z = np.array(size[grupo, "position_z_change_coord"].in_units("kpc"))
    VZ = np.array(size[grupo, "velocity_z"].in_units("km/s"))
    R = np.sqrt(X**2 + Y**2)
    d = {'ID':ID, 'X':X, 'VX':VX, 'Y':Y, 'VY':VY ,'Z': Z, 'VZ': VZ, "R":R}
    df = pd.DataFrame(data=d)
    return df

def data_to_csv_pre_gas (size, grupo_visualizar):
    X = np.array(size[grupo_visualizar,"position_x_change_coord"].in_units("kpc"))
    VX = np.array(size[grupo_visualizar, "velocity_x"].in_units("km/s"))
    Y = np.array(size[grupo_visualizar, "position_y_change_coord"].in_units("kpc"))
    VY = np.array(size[grupo_visualizar, "velocity_y"].in_units("km/s"))
    Z = np.array(size[grupo_visualizar, "position_z_change_coord"].in_units("kpc"))
    VZ = np.array(size[grupo_visualizar, "velocity_z"].in_units("km/s"))
    R = np.sqrt(X**2 + Y**2)
    d = {'X':X, 'VX':VX, 'Y':Y, 'VY':VY ,'Z': Z, 'VZ': VZ, "R":R}
    df = pd.DataFrame(data=d)
    return df


def apply_derived_field (ds, center, mean_v, matriz_rotacion):
    """This function applies all necessary derived fields for posterior analysis"""
    
    def _metallicity_2_gas(field,data):
        metalicidad = (data["gas","metal_ii_density"]+data["gas","metal_ia_density"])/data["gas","density"]
        return np.log10(metalicidad/0.0199)
    
    ds.add_field(("gas","metallicity_z"),function=_metallicity_2_gas,force_override=True, particle_type = False,
                 display_name='[z/H]',take_log=False,units="")
    
    def _metallicity_Fe_gas(field,data):
        metalicidad = (data["gas","metal_ia_density"])/data["gas","density"]
        return np.log10(metalicidad/0.00178)
    
    ds.add_field(("gas","metallicity_Fe"),function=_metallicity_Fe_gas,force_override=True,particle_type = False,
                  display_name='[Fe/H]',take_log=False,units="")
    
    
    def _metallicity_alpha_gas(field,data):
        metalicidad = (data["gas","metal_ia_density"])/data["gas","density"]
        return np.log10(metalicidad/0.0161)
    
    ds.add_field(("gas","metallicity_alpha"),function=_metallicity_alpha_gas,force_override=True,particle_type = False,
                  display_name='[alpha/H]',take_log=False,units="")
    
    def _metallicity_alphaFe_gas(field,data):
        metalicidad = data["gas","metal_ii_density"]/data["gas","metal_ia_density"]
        return (np.log10(metalicidad)-np.log10(0.0161/0.00178))
    
    ds.add_field(("gas","metallicity_alphaFe"),function=_metallicity_alphaFe_gas,force_override=True,particle_type = False,
                  display_name='[alpha/Fe]',take_log=False,units="")
    


    def _position_x_from_center_stars(field, data):
        return (data[("stars", "particle_position_x")] - center[0]).in_units('kpc')

    def _position_y_from_center_stars(field, data):
        return (data[("stars", "particle_position_y")] - center[1]).in_units('kpc')

    def _position_z_from_center_stars(field, data):
        return (data[("stars", "particle_position_z")] - center[2]).in_units('kpc')

    ds.add_field(("stars","position_x_from_center"), function=_position_x_from_center_stars, take_log=False, 
      particle_type="stars", units="cm", force_override = True,  sampling_type = "particle")
    ds.add_field(("stars","position_y_from_center"), function=_position_y_from_center_stars, take_log=False, 
         particle_type="stars", units="cm", force_override = True,  sampling_type = "particle")
    ds.add_field(("stars","position_z_from_center"), function=_position_z_from_center_stars, take_log=False, 
          particle_type="stars", units="cm", force_override = True,  sampling_type = "particle")

    def _position_x_from_center_dm(field, data):
        return (data[("darkmatter", "particle_position_x")] - center[0]).in_units('kpc')

    def _position_y_from_center_dm(field, data):
        return (data[("darkmatter", "particle_position_y")] - center[1]).in_units('kpc')

    def _position_z_from_center_dm(field, data):
        return (data[("darkmatter", "particle_position_z")] - center[2]).in_units('kpc')


    ds.add_field(("darkmatter","position_x_from_center"), function=_position_x_from_center_dm, take_log=False, 
      particle_type="darkmatter", units="cm", force_override = True,  sampling_type = "particle")
    ds.add_field(("darkmatter","position_y_from_center"), function=_position_y_from_center_dm, take_log=False, 
      particle_type="darkmatter", units="cm", force_override = True,  sampling_type = "particle")
    ds.add_field(("darkmatter","position_z_from_center"), function=_position_z_from_center_dm, take_log=False, 
     particle_type="darkmatter", units="cm", force_override = True,  sampling_type = "particle")



    def _position_x_change_coor_stars(field, data):
        x_inicial = data[("stars", "position_x_from_center")]
        y_inicial = data[("stars", "position_y_from_center")]
        z_inicial = data[("stars", "position_z_from_center")]
        x_change_coord= x_inicial*matriz_rotacion[0][0]+ y_inicial*matriz_rotacion[0][1] + z_inicial*matriz_rotacion[0][2]
        return x_change_coord

    def _position_y_change_coor_stars(field, data):
        x_inicial = data[("stars", "position_x_from_center")]
        y_inicial = data[("stars", "position_y_from_center")]
        z_inicial = data[("stars", "position_z_from_center")]
        y_change_coord= x_inicial*matriz_rotacion[1][0]+ y_inicial*matriz_rotacion[1][1] + z_inicial*matriz_rotacion[1][2]
        return y_change_coord

    def _position_z_change_coor_stars(field, data):
        x_inicial = data[("stars", "position_x_from_center")]
        y_inicial = data[("stars", "position_y_from_center")]
        z_inicial = data[("stars", "position_z_from_center")]
        z_change_coord= x_inicial*matriz_rotacion[2][0]+ y_inicial*matriz_rotacion[2][1] + z_inicial*matriz_rotacion[2][2]
        return z_change_coord


    ds.add_field(("stars","position_x_change_coord"), function=_position_x_change_coor_stars, take_log=False, 
        particle_type="stars", units="cm", force_override = True, sampling_type = "particle")
    ds.add_field(("stars","position_y_change_coord"), function=_position_y_change_coor_stars, take_log=False,
        particle_type="stars", units="cm", force_override = True,  sampling_type = "particle")
    ds.add_field(("stars","position_z_change_coord"), function=_position_z_change_coor_stars, take_log=False,
        particle_type="stars", units="cm", force_override = True,  sampling_type = "particle")

    def _position_x_change_coor_dm(field, data):
        x_inicial = data[("darkmatter", "position_x_from_center")]
        y_inicial = data[("darkmatter", "position_y_from_center")]
        z_inicial = data[("darkmatter", "position_z_from_center")]
        x_change_coord= x_inicial*matriz_rotacion[0][0]+ y_inicial*matriz_rotacion[0][1] + z_inicial*matriz_rotacion[0][2]
        return x_change_coord

    def _position_y_change_coor_dm(field, data):
        x_inicial = data[("darkmatter", "position_x_from_center")]
        y_inicial = data[("darkmatter", "position_y_from_center")]
        z_inicial = data[("darkmatter", "position_z_from_center")]
        y_change_coord= x_inicial*matriz_rotacion[1][0]+ y_inicial*matriz_rotacion[1][1] + z_inicial*matriz_rotacion[1][2]
        return y_change_coord

    def _position_z_change_coor_dm(field, data):
        x_inicial = data[("darkmatter", "position_x_from_center")]
        y_inicial = data[("darkmatter", "position_y_from_center")]
        z_inicial = data[("darkmatter", "position_z_from_center")]
        z_change_coord= x_inicial*matriz_rotacion[2][0]+ y_inicial*matriz_rotacion[2][1] + z_inicial*matriz_rotacion[2][2]
        return z_change_coord

    ds.add_field(("darkmatter","position_x_change_coord"), function=_position_x_change_coor_dm, take_log=False, 
        particle_type="darkmatter", units="cm", force_override = True, sampling_type = "particle")
    ds.add_field(("darkmatter","position_y_change_coord"), function=_position_y_change_coor_dm, take_log=False,
        particle_type="darkmatter", units="cm", force_override = True,  sampling_type = "particle")
    ds.add_field(("darkmatter","position_z_change_coord"), function=_position_z_change_coor_dm, take_log=False,
        particle_type="darkmatter", units="cm", force_override = True,  sampling_type = "particle")

  
  
    def _velocity_x_change_coor_stars(field, data):
        x_inicial = data[("stars", "particle_velocity_x")] - mean_v[0]
        y_inicial = data[("stars", "particle_velocity_y")] - mean_v[1]
        z_inicial = data[("stars", "particle_velocity_z")] - mean_v[2]
        x_change_coord= x_inicial*matriz_rotacion[0][0]+ y_inicial*matriz_rotacion[0][1] + z_inicial*matriz_rotacion[0][2]
        return x_change_coord

    def _velocity_y_change_coor_stars(field, data):
        x_inicial = data[("stars", "particle_velocity_x")] - mean_v[0]
        y_inicial = data[("stars", "particle_velocity_y")] - mean_v[1]
        z_inicial = data[("stars", "particle_velocity_z")] - mean_v[2]
        x_change_coord= x_inicial*matriz_rotacion[1][0]+ y_inicial*matriz_rotacion[1][1] + z_inicial*matriz_rotacion[1][2]
        return x_change_coord
    def _velocity_z_change_coor_stars(field, data):
        x_inicial = data[("stars", "particle_velocity_x")] - mean_v[0]
        y_inicial = data[("stars", "particle_velocity_y")] - mean_v[1]
        z_inicial = data[("stars", "particle_velocity_z")] - mean_v[2]
        z_change_coord= x_inicial*matriz_rotacion[2][0]+ y_inicial*matriz_rotacion[2][1] + z_inicial*matriz_rotacion[2][2]
        return z_change_coord

  
    ds.add_field(("stars","velocity_x"), function=_velocity_x_change_coor_stars, take_log=False, 
     particle_type="stars", units="km/s", force_override = True,  sampling_type = "particle")
    ds.add_field(("stars","velocity_y"), function=_velocity_y_change_coor_stars, take_log=False, 
     particle_type="stars", units="km/s", force_override = True,  sampling_type = "particle")
    ds.add_field(("stars","velocity_z"), function=_velocity_z_change_coor_stars, take_log=False, 
       particle_type="stars", units="km/s", force_override = True,  sampling_type = "particle")

       
    def _velocity_x_change_coor_dm(field, data):
        x_inicial = data[("darkmatter", "particle_velocity_x")]
        y_inicial = data[("darkmatter", "particle_velocity_y")]
        z_inicial = data[("darkmatter", "particle_velocity_z")]
        x_change_coord= x_inicial*matriz_rotacion[0][0]+ y_inicial*matriz_rotacion[0][1] + z_inicial*matriz_rotacion[0][2]
        return x_change_coord

    def _velocity_y_change_coor_dm(field, data):
        x_inicial = data[("darkmatter", "particle_velocity_x")]
        y_inicial = data[("darkmatter", "particle_velocity_y")]
        z_inicial = data[("darkmatter", "particle_velocity_z")]
        x_change_coord= x_inicial*matriz_rotacion[1][0]+ y_inicial*matriz_rotacion[1][1] + z_inicial*matriz_rotacion[1][2]
        return x_change_coord
    def _velocity_z_change_coor_dm(field, data):
        x_inicial = data[("darkmatter", "particle_velocity_x")]
        y_inicial = data[("darkmatter", "particle_velocity_y")]
        z_inicial = data[("darkmatter", "particle_velocity_z")]
        z_change_coord= x_inicial*matriz_rotacion[2][0]+ y_inicial*matriz_rotacion[2][1] + z_inicial*matriz_rotacion[2][2]
        return z_change_coord

    ds.add_field(("darkmatter","velocity_x"), function=_velocity_x_change_coor_dm, take_log=False, 
      particle_type="darkmatter", units="km/s", force_override = True,  sampling_type = "particle")
    ds.add_field(("darkmatter","velocity_y"), function=_velocity_y_change_coor_dm, take_log=False, 
      particle_type="darkmatter", units="km/s", force_override = True,  sampling_type = "particle")
    ds.add_field(("darkmatter","velocity_z"), function=_velocity_z_change_coor_dm, take_log=False, 
       particle_type="darkmatter", units="km/s", force_override = True,  sampling_type = "particle")
    

    ds.add_field(("stars","particle_AlphaH"), function=metallicity_star_alfaH, take_log=False, particle_type=True, units="")
    ds.add_field(("stars","particle_FeH"), function=metallicity_star_FeH, take_log=False, particle_type=True, units="")
    ds.add_field(("stars","particle_AlphaFe"), function=metallicity_star_AlfaFe, take_log=False, particle_type=True, units="")

    def _age(field, data):
        return data.ds.current_time.in_units("Gyr") - data[ "stars","particle_creation_time"].in_units("Gyr")
    ds.add_field(("stars","age"), function=_age, take_log=True, particle_type="stars", units="Gyr", force_override = True)

    #ds.add_field(("stars","age"), function=age, take_log=True, particle_type="stars", units="Gyr", force_override = True)

    def _gas_position_x_from_center(field, data):
        return (data[("gas", "x")] - center[0]).in_units('kpc')


    def _gas_position_y_from_center(field, data):
        return (data[("gas", "y")] - center[1]).in_units('kpc')


    def _gas_position_z_from_center(field, data):
        return (data[("gas", "z")] - center[2]).in_units('kpc')


    ds.add_field(("gas","position_y_from_center"), function=_gas_position_y_from_center, take_log=False, units="cm", force_override = True)
    ds.add_field(("gas","position_x_from_center"), function=_gas_position_x_from_center, take_log=False,units="cm", force_override = True)
    ds.add_field(("gas","position_z_from_center"), function=_gas_position_z_from_center, take_log=False, units="cm", force_override = True)

    def _gas_position_x_change_coor(field, data):
        x_inicial = data[("gas", "position_x_from_center")]
        y_inicial = data[("gas", "position_y_from_center")]
        z_inicial = data[("gas", "position_z_from_center")]
        x_change_coord= x_inicial*matriz_rotacion[0][0]+ y_inicial*matriz_rotacion[0][1] + z_inicial*matriz_rotacion[0][2]
        return x_change_coord

    def _gas_position_y_change_coor(field, data):
        x_inicial = data[("gas", "position_x_from_center")]
        y_inicial = data[("gas", "position_y_from_center")]
        z_inicial = data[("gas", "position_z_from_center")]
        y_change_coord= x_inicial*matriz_rotacion[1][0]+ y_inicial*matriz_rotacion[1][1] + z_inicial*matriz_rotacion[1][2]
        return y_change_coord

    def _gas_position_z_change_coor(field, data):
        x_inicial = data[("gas", "position_x_from_center")]
        y_inicial = data[("gas", "position_y_from_center")]
        z_inicial = data[("gas", "position_z_from_center")]
        z_change_coord= x_inicial*matriz_rotacion[2][0]+ y_inicial*matriz_rotacion[2][1] + z_inicial*matriz_rotacion[2][2]
        return z_change_coord

    ds.add_field(("gas","position_x_change_coord"), function=_gas_position_x_change_coor, take_log=False, units="cm", force_override = True)
    ds.add_field(("gas","position_y_change_coord"), function=_gas_position_y_change_coor, take_log=False, units="cm", force_override = True)
    ds.add_field(("gas","position_z_change_coord"), function=_gas_position_z_change_coor, take_log=False,  units="cm", force_override = True)
    
    def _gas_velocity_z_change_coor(field, data):
        x_inicial = data[("gas", "velocity_x")]-mean_v[0]
        y_inicial = data[("gas", "velocity_y")]- mean_v[1]
        z_inicial = data[("gas", "velocity_z")]- mean_v[2]
        z_change_coord= x_inicial*matriz_rotacion[2][0]+ y_inicial*matriz_rotacion[2][1] + z_inicial*matriz_rotacion[2][2]
        return z_change_coord
                         
    
    
    def _gas_velocity_x_change_coor(field, data):
        x_inicial = data[("gas", "velocity_x")]- mean_v[0]
        y_inicial = data[("gas", "velocity_y")]- mean_v[1]
        z_inicial = data[("gas", "velocity_z")]- mean_v[2]
        x_change_coord= x_inicial*matriz_rotacion[0][0]+ y_inicial*matriz_rotacion[0][1] + z_inicial*matriz_rotacion[0][2]
        return x_change_coord
                         
    
    
    def _gas_velocity_y_change_coor(field, data):
        x_inicial = data[("gas", "velocity_x")]-mean_v[0]
        y_inicial = data[("gas", "velocity_y")]-mean_v[1]
        z_inicial = data[("gas", "velocity_z")]-mean_v[2]
        x_change_coord= x_inicial*matriz_rotacion[1][0]+ y_inicial*matriz_rotacion[1][1] + z_inicial*matriz_rotacion[1][2]
        return x_change_coord
                         
    
    ds.add_field(("gas","gas_velocity_z"), function=_gas_velocity_z_change_coor, take_log=False, 
                 units="cm/s",  force_override = True)    
       
    ds.add_field(("gas","gas_velocity_x"), function=_gas_velocity_x_change_coor, take_log=False, 
                  units="cm/s",  force_override = True)
        
    ds.add_field(("gas","gas_velocity_y"), function=_gas_velocity_y_change_coor, take_log=False, 
                  units="cm/s", force_override = True)
    

    return ds
