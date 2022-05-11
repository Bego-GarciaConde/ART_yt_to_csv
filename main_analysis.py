import yt
from yt import YTArray

from yt.utilities.cosmology import Cosmology

co = Cosmology(hubble_constant=0.7, omega_matter=0.3,
               omega_lambda=0.7, omega_curvature=0.0)

import numpy as np
from yt.units import G
import array
import pandas as pd


#from skspatial.objects import Point, Vector, Plane
#from skspatial.plotting import plot_3d

import gc
from yt_derived_field import *
from useful_functions import *
from disk_particles import *
from READ_yt_to_CSV import *

from config import *

def main_analysis(name, path_snapshots):
    ds = yt.load(path_snapshots + f"20MpcBox_HartGalMW003_RUN2.2_a0.{name}.d")
    ct = float(ds.current_time.in_units("Gyr"))
    lb = 13.97 - ct
    # ad = ds.all_data()
    
    
    if center_rvir_calc ==1:
        print("Calculating center and Rvir")
        center = calculate_center(ds)
        Rvir = calculate_rvir(ds, center ,delta_vir)
        cx, cy, cz = center[0].in_units("cm"), center[1].in_units("cm"), center[2].in_units("cm")
        arr = array.array('f', [cx,cy,cz, Rvir])
        np.savetxt(path_datos +"center_%d.txt"  % name, arr, delimiter=' ')
    else:
        centro = np.loadtxt(path_datos +f'center_{name}.txt')
        center = YTArray([centro[0], centro[1], centro[2]], "cm")
        cx,cy,cz = center[0].in_units("cm"), center[1].in_units("cm"),  center[2].in_units("cm")
        Rvir = YTArray(centro[3], "kpc")

  

    yt.add_particle_filter("young_stars",function=young_star_filter,filtered_type="stars",requires=["particle_creation_time"])
    ds.add_particle_filter("young_stars")

    print("Calculating angular momentum and mean velocity")
    sp5 = ds.sphere(center, (0.1*Rvir))  #Sphere of 10% Virial Radius
    angular_momentum = calculate_angular_momentum (ds, center, Rvir)
  #  angular_momentum = YTArray([2.63425158e+29, 4.18786719e+28, -1.09858375e+29], "cm**2/s")
    mean_vx, mean_vy, mean_vz = calculate_mean_velocity (sp5)
    mean_v = [mean_vx, mean_vy, mean_vz]
    matriz_rotacion = calculate_rotation_matrix (angular_momentum)

    print("Applying derived field")

   #----FIRST ALIGNMENT-------
    ds = apply_derived_field (ds, center, mean_v, matriz_rotacion)
    size=ds.sphere(center,Rvir)
    
    #----SECOND ALIGNMENT-----
    print("Re-centering")
    df = data_to_csv_pre(size, "stars")
    ind=(df['R']<15.)&(df['Z']<4)&(df["Z"]>-4)
    pos0=df[['X','Y','Z']].to_numpy()
    vel0=df[['VX','VY','VZ']].to_numpy()
    B = centering(pos0,vel0,ind,means=True,medians=False,L=True)
    #B =np.zeros((3,3))
    print("saving data...")

    if save_stars == 1:
        print("Saving star data")
        X,Y,Z, = apply_transformation_matrix(B,df["X"] ,df["Y"],df["Z"])
        VX, VY, VZ = apply_transformation_matrix(B,df["VX"] ,df["VY"],df["VZ"])
      #  X,Y,Z, = df["X"] ,df["Y"],df["Z"]
      #  VX, VY, VZ = df["VX"] ,df["VY"],df["VZ"]
        Phi =  np.mod(np.arctan2(Y,X), 2*np.pi)
        R = np.sqrt(X**2 + Y**2)
        Vphi = (-X*VY + Y*VX)/R #todo revisar signos de phi y vphi
        VR = (X*VX + Y*VY)/R

        age_stars = np.array(size["stars", "age"].in_units("Myr"))
        AlphaH = np.array(size["stars", "particle_AlphaH"])
        FeH = np.array(size["stars", "particle_FeH"])
        AlphaFe = np.array(size["stars", "particle_AlphaFe"])
        mass_stars = np.array(size["stars", "particle_mass"].in_units("Msun"))


        d = {'ID':np.array(df["ID"]), 'X':X, 'VX':VX, 'Y':Y, 'VY':VY ,'Z': Z, 'VZ': VZ, 'Phi':Phi, 'Vphi' : Vphi, 'R': R,
            'Vr': VR,"Age":age_stars, "Mass": mass_stars, "AlphaH": AlphaH, "FeH":FeH, "AlphaFe": AlphaFe }
      #  d = {'ID':np.array(df["ID"]), 'X':X, 'VX':VX, 'Y':Y, 'VY':VY ,'Z': Z, 'VZ': VZ, "Age":age_stars, "Mass": mass_stars}
        df = pd.DataFrame(data=d, dtype = np.float32)
        buf = path_guardado_snapshots + f"{name}_stars_Rvir.csv" 
        df.to_csv(buf,  sep = ",",index = False, float_format='%.3f')

    if save_dm == 1:
        grupo_visualizar = "darkmatter"
        df = data_to_csv_pre(size, grupo_visualizar)
        X,Y,Z, = apply_transformation_matrix(B,df["X"] ,df["Y"],df["Z"])
        VX, VY, VZ = apply_transformation_matrix(B,df["VX"] ,df["VY"],df["VZ"])
        mass_darkmatter = np.array(size[grupo_visualizar, "particle_mass"].in_units("Msun"))

        R = np.sqrt(X**2 + Y**2)
        d = {'ID':np.array(df["ID"]), 'X':X, 'VX':VX, 'Y':Y, 'VY':VY ,'Z': Z,  'VZ': VZ, "Mass": mass_darkmatter }
        df = pd.DataFrame(data=d, dtype = np.float32)
        buf = path_guardado_snapshots + f"{name}_dm_Rvir.csv" 
        df.to_csv(buf,  sep = ",",index = False,float_format= "%.3f")

    if save_gas==1:
        grupo_visualizar = "gas"
        df = data_to_csv_pre_gas(size, grupo_visualizar)

        X,Y,Z, = apply_transformation_matrix(B,df["X"] ,df["Y"],df["Z"])
        VX, VY, VZ = apply_transformation_matrix(B,df["VX"] ,df["VY"],df["VZ"])
        Phi =  np.mod(np.arctan2(Y,X), 2*np.pi)
        R = np.sqrt(X**2 + Y**2)
        Vphi = (-X*VY + Y*VX)/R
        VR = (X*VX + Y*VY)/R
        distancia = np.sqrt(X**2 + Y**2 + Z**2)
        #------------
        temperature = np.array(size["gas", "temperature"])
        n_H = np.array(size["gas", "H_nuclei_density"])
        masa = np.array(size["gas", "cell_mass"].in_units("Msun"))
        #-------------
        masa_acumulada = gas_masa_acumulada(size, distancia)
        #-------------  
        d = {'X':X, 'VX':VX, 'Y':Y, 'VY':VY ,'Z': Z, 'VZ': VZ, 'Phi':Phi, 'Vphi' : Vphi, 
             'R': R,'Vr': VR,"Temperature":temperature, "Mass": masa,
            "Distance":distancia,"Masa_acumulada": masa_acumulada,"nH": n_H }
        df = pd.DataFrame(data=d, dtype = np.float32)
        df.to_csv(path_guardado_snapshots + f"Gas_{name}.csv",  sep=',', index = False)

    if save_disk_particles==1:
        particles_disk(name, path_snapshots, path_datos)
    gc.collect()

    