import yt
from yt import YTArray

from yt.utilities.cosmology import Cosmology

co = Cosmology(hubble_constant=0.7, omega_matter=0.3, 
               omega_lambda=0.7, omega_curvature=0.0)

import numpy as np 

import pandas as pd


from os.path import expanduser
home = expanduser("~")
from mpl_toolkits.axes_grid1 import make_axes_locatable

from yt_derived_field import *
from useful_functions import *
from main_analysis import *
from config import *
#path_datos = "/home/bego/GARROTXA/datos_GARROTXA_resim/"
import multiprocessing
from multiprocessing import Pool


#name = 728
def particles_disk(name, path_snapshots, path_datos):
    data =path_snapshots + "20MpcBox_HartGalMW003_RUN2.2_a0.%s.d" %name
    centro = np.loadtxt(path_datos + 'center_{}.txt'.format(name)) 
    center = YTArray([centro[0], centro[1], centro[2]], "cm")
    Rvir = YTArray(centro[3], "kpc")

    ds = yt.load(data)

    GasMassType_to_use = "cell_mass"
    
    minimo = 2e20
    maximo = 1e23
    bins_perfil = 85
    
    sp = ds.sphere(center, (43,'kpc'))
    # CREATE MASS PROFILE TO CALCULATE ACCUMULATED MASS 
    profile_part= yt.create_profile(sp, ('stars',"particle_position_spherical_radius"), ('stars','particle_mass'), n_bins=bins_perfil, weight_field = None,extrema={('stars',"particle_position_spherical_radius"): (minimo,maximo)}, accumulation= True)
    binnum=len(profile_part.x)
    profile_DM = yt.create_profile(sp, ('darkmatter',"particle_position_spherical_radius"), ('darkmatter','particle_mass'), n_bins=bins_perfil, weight_field = None,extrema={('darkmatter',"particle_position_spherical_radius"): (minimo, maximo)},accumulation = True)
    profile_gas = yt.create_profile(sp, "radius", GasMassType_to_use, weight_field =None, n_bins=binnum,extrema={'radius': (minimo, maximo)}, accumulation = True)

    masa_perfil=[0.0]*len(profile_part.x)
    masa_perfil[0]=(profile_part["stars","particle_mass"][0].in_units("Msun").value+ profile_DM["darkmatter","particle_mass"][0].in_units("Msun").value+profile_gas[GasMassType_to_use][0].in_units("Msun").value)
    for i in range(1,len(masa_perfil)):
        masa_perfil[i]=(profile_part["stars","particle_mass"][i].in_units("Msun").value+ profile_DM["darkmatter","particle_mass"][i].in_units("Msun").value+profile_gas[GasMassType_to_use][i].in_units("Msun").value)
    del sp
  
    radio_masa= np.array(profile_part.x.in_units("kpc"))
    masa_perfil_2 = np.round(np.array(masa_perfil), 1)

  
    def asignacion_masa(radio):
        for i in range(0,len(radio_masa)):
            if radio <radio_masa[i]:
                masa = (masa_perfil_2[i-1 ] + masa_perfil_2[i])/2
                return masa
                break
    

    yt.add_particle_filter("young_stars",function=young_star_filter,filtered_type="stars",requires=["particle_creation_time"])
    ds.add_particle_filter("young_stars")

 
    sp5 = ds.sphere(center, (0.1*Rvir))  #Sphere of 10% Virial Radius
    angular_momentum = calculate_angular_momentum (ds, center, Rvir)
    mean_vx, mean_vy, mean_vz = calculate_mean_velocity (sp5)
    mean_v = [mean_vx, mean_vy, mean_vz]
    matriz_rotacion = calculate_rotation_matrix (angular_momentum) 

    ds = apply_derived_field (ds, center, mean_v, matriz_rotacion)

    print("Defining spheres...")

  #  disco_analisis = ds.disk(center, angular_momentum, (25, "kpc"),(10, "kpc"))
    size =  ds.sphere(center, (25, "kpc")) 
    grupo_visualizar = "stars"
    
    df = data_to_csv_pre(size, "stars")
    ind=(df['R']<15.)&(df['Z']<4)&(df["Z"]>-4)
    pos0=df[['X','Y','Z']].to_numpy()
    vel0=df[['VX','VY','VZ']].to_numpy()
    B = centering(pos0,vel0,ind,means=True,medians=False,L=True)

    X,Y,Z, = apply_transformation_matrix(B,df["X"] ,df["Y"],df["Z"])
    VX, VY, VZ = apply_transformation_matrix(B,df["VX"] ,df["VY"],df["VZ"])
    Phi =  np.mod(np.arctan2(Y,X), 2*np.pi)
    R = np.sqrt(X**2 + Y**2)
    Vphi = (-X*VY + Y*VX)/R
    VR = (X*VX + Y*VY)/R

    star_id = np.array (size[grupo_visualizar, 'particle_index'].in_units("dimensionless"))
    age = np.array(size[grupo_visualizar, "age"].in_units("Gyr"))
    radio_centro = np.array(size[grupo_visualizar, "particle_position_spherical_radius"].in_units("kpc"))
    mass = np.array(size[grupo_visualizar, "particle_mass"].in_units("Msun"))

    
    G = 132703693031 # G IN UNITS km**3/(Msun*s**2) 
    del size 
    
    masa_acumulada = []
    velocidad_circular = []
   
    for i in range(len (radio_centro)):
        masa= asignacion_masa(radio_centro[i])
        masa_acumulada.append(masa)
        vel = np.sqrt((G*masa)/(radio_centro[i]*3.086e+16))
        velocidad_circular.append(vel)
    


    jz_jc = Vphi/velocidad_circular

    a = (VZ*Y - VY*Z)**2
    b = (VX*Z - VZ*X)**2
    c = (VY*X - VX*Y)**2

    cos_alpha = Vphi*R /(a + b + c)


    d = {'ID':star_id,'Distancia': radio_centro,
     'Vcirc':velocidad_circular , 'JzJc': jz_jc, 'cos_alpha': cos_alpha, "Z":Z}
    df = pd.DataFrame(data=d, dtype = np.float32)
    

    
    #Clasificación de estrelas que están en el disco 
    #condiciones geométricas
    df_class0 = df[df["Distancia"]< 25 ].copy()
    #kinematic conditions
    df_class1 = df_class0[(df_class0["cos_alpha"]>0) & (df_class0["cos_alpha"]<0.7) ].copy()
    df_disco = df_class1[(df_class1["JzJc"]<1.5)& (df_class1["JzJc"]> 0.5) &(np.abs(df_class1["Z"])< 5) ].copy()

    buf = "/media/temp/bego/disco/Stars_disco_%d.csv"  % name
    df_disco.to_csv(buf,  sep=',')
    

    #df_disco = pd.DataFrame(data=datos)

    ##SURFACE DENSITY
    #
    #radios = np.arange(1, 25.5, 0.5).tolist()
    #print(radios)
    #
    #masa_r = []
    #area_r=[]
    #
    #for i in range(len(radios)):
    #    masa_i = 0
    #    print (radios[i])
    #    for j in range (len(df_disco)):
    #        if df_disco["Distancia"][j] <= radios[i]:
    #            masa_i = masa_i + df_disco["Mass"][j]
    #    print(masa_i)
    #    area_r_i = np.pi*radios[i]**2
    #    print(area_r_i)
    #    masa_r.append(masa_i)
    #    area_r.append(area_r_i)
    #
    #
    #area_anillo=[]
    #area_anillo.append(area_r[0])
    #masa_anillo = []
    #masa_anillo.append(masa_r[0])
    #
    #for i in range(1, len(area_r)):
    #    masa_an = masa_r[i] - masa_r[i-1]
    #    print(masa_an)
    #    area_an = area_r[i] - area_r[i-1]   
    #    print(area_an)
    #    masa_anillo.append(masa_an)
    #    area_anillo.append(area_an)
    #
    #print("Masa del anillo")
    #print(masa_anillo)
    #print("Area del anillo")
    #print(area_anillo)
    #densidad_sup = []
    #for i in range(len (masa_anillo)):
    #    densidad = (masa_anillo[i])/area_anillo[i]
    #    densidad_sup.append(densidad)
    #
    #print("Densidad superficial")
    #print(densidad_sup)
    #
    #
    #p = plt.figure()
    #
    #plt.plot(radios, np.log10(densidad_sup), '-', color = 'black')
    #
    #plt.ylabel ('log $\Sigma_{d}$($M_{\odot}/kpc^2$)')
    #plt.xlabel ('R (kpc)')
    #plt.title ("Surface density distribution")
    #p.savefig ('Densidad_sup.png', dpi = 400)
    #
    #
    #data_surface_density = {'Rad':radios, 'Density': densidad_sup}
    #df_surface_density = pd.DataFrame(data=data_surface_density)
    #buf = "surface_density_stars_%d.txt"  % name
    #df_surface_density.to_csv(buf,  sep='\t')

