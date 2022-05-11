import yt
import math
from yt import YTArray

from yt.utilities.cosmology import Cosmology

co = Cosmology(hubble_constant=0.7, omega_matter=0.3, 
               omega_lambda=0.7, omega_curvature=0.0)

import numpy as np 
from yt.units import G 
import os
import array


import matplotlib
from scipy.interpolate import RectBivariateSpline
import pandas as pd
import matplotlib.colors as mcolors
import os
from scipy import stats

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec

from matplotlib import rcParams
from yt.utilities.cosmology import Cosmology
from skspatial.objects import Point, Vector, Plane

def centering(pos0,vel0,ind,means=True,medians=False,L=True):
    pos=pos0.copy()
    vel=vel0.copy()
    
    pos2=pos0.copy()
    vel2=vel0.copy()
    
    #centering particles with means   
    if means==True:
        for j in range(3):
            mp=np.mean(pos0[ind,j])
            mv=np.mean(vel0[ind,j])
            print(j,mp,mv)
            pos[:,j]=pos0[:,j]-mp
            vel[:,j]=vel0[:,j]-mv

    #centering particles with medians   
    if medians==True:
        for j in range(3):
            pos[:,j]=pos0[:,j]-np.median(pos0[ind,j])
            vel[:,j]=vel0[:,j]-np.median(vel0[ind,j])    
    
    #reorienting particles so meadian L is oriented as Lz (Lx=Ly=0) 
    # the other free axis X is taken so as to produce the minimum variation of the old axis X
    if L==True:    
        Lx=pos[:,1]*vel[:,2]-pos[:,2]*vel[:,1]
        Ly=pos[:,2]*vel[:,0]-pos[:,0]*vel[:,2]
        Lz=pos[:,0]*vel[:,1]-pos[:,1]*vel[:,0]
       #finding median L vector only with selected particles
        mLx,mLy,mLz= np.mean(Lx[ind]),np.mean(Ly[ind]),np.mean(Lz[ind])
   
        m=np.sqrt(mLx*mLx+mLy*mLy+mLz*mLz)
#        print('0',np.round((mLx,mLy,mLz),2))
        mLx,mLy,mLz=-mLx/m,-mLy/m,-mLz/m #normalization of the median L vector         
        v3=[mLx,mLy,mLz]

        plane = Plane(point=[0, 0, 0], normal=[mLx,mLy,mLz])
        point = Point([1, 0, 0])
        v1 = plane.project_point(point) 
        m=np.sqrt(v1[0]**2+v1[1]**2+v1[2]**2)
        v1=v1/m

        v2=np.cross(v3,v1)
        m=np.sqrt(v2[0]**2+v2[1]**2+v2[2]**2)
        v2=v2/m           
        A=np.matrix([np.array(v1),np.array(v2),np.array(v3)]).T

        B=np.linalg.inv(A)

    return B
 

def calculate_rotation_matrix(angular_momentum):
    #this was made by trial and error. I don't have any idea how this works
    angulo_z = np.arctan2(angular_momentum[0],angular_momentum[1])
    theta_z = 0

    angulo_x = np.arctan2(angular_momentum[1],angular_momentum[2])
    theta_x = angulo_x

    angulo_y = np.arctan2(angular_momentum[0],angular_momentum[2])
    theta_y =   angulo_y + np.pi

    Rx = np.array([[1, 0, 0],
        [0, math.cos(theta_x), -math.sin(theta_x)],
        [0, math.sin(theta_x),math.cos(theta_x)]])

    Ry = np.array([[math.cos(theta_y), 0, math.sin(theta_y)],
        [0, 1, 0],
        [-math.sin(theta_y), 0,math.cos(theta_y)]])

    Rz = np.array([[math.cos(theta_z), -math.sin(theta_z),0],
        [math.sin(theta_z), math.cos(theta_z), 0],
        [0, 0,1]])

    res = np.dot(Ry,Rx)  #First rotate y axis and then x axis

    matriz_rotacion =res
    return matriz_rotacion

def apply_transformation_matrix(matriz_trans, X,Y,Z):
    X_re= X*matriz_trans[0,0]+ Y*matriz_trans[0,1] + Z*matriz_trans[0,2]       
    Y_re= X*matriz_trans[1,0]+ Y*matriz_trans[1,1] + Z*matriz_trans[1,2]
    Z_re= X*matriz_trans[2,0]+ Y*matriz_trans[2,1] + Z*matriz_trans[2,2]
    return X_re, Y_re, Z_re


def gas_masa_acumulada(sp, distancia):
    GasMassType_to_use = "cell_mass"
    minimo = 2e20
    maximo = 5e23
    bins_perfil = 120

    # CREATE MASS PROFILE TO CALCULATE ACCUMULATED MASS
    profile_part= yt.create_profile(sp, ('stars',"particle_position_spherical_radius"), ('stars','particle_mass'), n_bins=bins_perfil, weight_field = None,extrema={('stars',"particle_position_spherical_radius"): (minimo,maximo)}, accumulation= True)
    binnum=len(profile_part.x)
    profile_DM = yt.create_profile(sp, ('darkmatter',"particle_position_spherical_radius"), ('darkmatter','particle_mass'), n_bins=bins_perfil, weight_field = None,extrema={('darkmatter',"particle_position_spherical_radius"): (minimo, maximo)},accumulation = True)
    profile_gas = yt.create_profile(sp, "radius", GasMassType_to_use, weight_field =None, n_bins=binnum,extrema={'radius': (minimo, maximo)}, accumulation = True)
    masa_perfil=(np.array(profile_part["stars","particle_mass"].in_units("Msun"))+ np.array(profile_DM["darkmatter","particle_mass"].in_units("Msun"))+
                np.array(profile_gas[GasMassType_to_use].in_units("Msun")))

    radio_masa= np.array(profile_part.x.in_units("kpc"))
    masa_perfil_2= np.around(masa_perfil, decimals = 1)

    def asignacion_masa(radio):
        for i in range(0,len(radio_masa)):
            if radio <radio_masa[i]:
                masa = (masa_perfil_2[i-1 ] + masa_perfil_2[i])/2
                return masa
                break
    
    masa_acumulada = []
    # distancia_dig = np.digitize(distancia,bins=len(masa_perfil)-1)
    # for i in range(len(distancia_dig)):
    #     dis = distancia_dig[i]
    #     masa_acumulada.append(masa_perfil_2[dis])
 
    for i in range(len(distancia)):
        dis = distancia[i]
        masa_acu = asignacion_masa(dis)
        masa_acumulada.append(masa_acu)
        
    return masa_acumulada