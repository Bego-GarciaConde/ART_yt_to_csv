# -*- coding: utf-8 -*-
#
"""
Created on 24/01/2022
@author: B. Garcia-Conde
"""
import gc

from yt_derived_field import *
from useful_functions import *
from main_analysis import *
from config import *
#from disk_particles import particles_disk

#label of the snapshots a=0.xxx


#snapshots_analysis = [886]

def main():
    for name in snapshots_analysis:
       # name = snapshots_analysis[i]
        if name < 425:
            path_snapshots = "/media/temp1/bego/GARROTXA_ART/"
        elif (name >= 425)&(name < 600):
            path_snapshots = "/srv/cab1/garrotxa/GARROTXA_ART/MW_003/RUN2.2/"
        elif (name >=600 )&(name < 800):
            path_snapshots = "/home/Garrotxa_ART/New_Run/"
        elif (name >= 800) & (name < 900) :
            path_snapshots = "/media/temp/bego/New_Resim/"
        elif name >= 900 :
            path_snapshots = "/media/temp1/GARROTXA_ART/MW_003/RUN2.2/"

        print("Loading data")
        #data = yt.load(path_snapshots + "20MpcBox_HartGalMW003_RUN2.2_a0.%s.d" %name)

        main_analysis(name, path_snapshots)
    #    particles_disk(name, path_snapshots)
        gc.collect()
        
if __name__ == "__main__":

    main()