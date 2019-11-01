#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:28:53 2019

@author: dminh
"""
import IO
import numpy as np

def center_of_mass(inpcrd_file_name, prmtop_file_name):
    """
    return the center of mass of self._crd
    """
    prmtop = dict()
    prmtop = IO.PrmtopLoad(prmtop_file_name).get_parm_for_grid_calculation()
    center_of_mass = np.zeros([3], dtype=float)
    masses = prmtop["MASS"]
    crd_temp = IO.InpcrdLoad(inpcrd_file_name).get_coordinates()
    natoms = prmtop["POINTERS"]["NATOM"]
    if (crd_temp.shape[0] != natoms) or (crd_temp.shape[1] != 3):
        raise RuntimeError("coordinates in %s has wrong shape"%inpcrd_file_name)
    for atom_ind in range(len(crd_temp)):
        center_of_mass += masses[atom_ind] * crd_temp[atom_ind]
    total_mass = masses.sum()
    if total_mass == 0:
        raise RuntimeError("zero total mass")
    return center_of_mass / total_mass

def displacement(Rec_prmtop, Rec_crd, Lig_prmtop, Lig_crd):
    Lig_R=center_of_mass(Lig_crd, Lig_prmtop)
    Rec_R=center_of_mass(Rec_crd, Rec_prmtop)
    return (Lig_R - Rec_R)

if __name__ == "__main__":
    Lig_crd="/home/chamila/test-3/2.redock/1.amber/1A2K_C:AB/ligand.inpcrd"
    Lig_prmtop="/home/chamila/test-3/2.redock/1.amber/1A2K_C:AB/ligand.prmtop"
    Lig_R=center_of_mass(Lig_crd, Lig_prmtop)
    print("Lig center of mass=", Lig_R)

    Rec_crd="/home/chamila/test-3/2.redock/1.amber/1A2K_C:AB/receptor.inpcrd"
    Rec_prmtop="/home/chamila/test-3/2.redock/1.amber/1A2K_C:AB/receptor.prmtop"
    Rec_R=center_of_mass(Rec_crd, Rec_prmtop)
    print("Rec center of mass=", Rec_R)


    Native_displacement = displacement(Rec_prmtop, Rec_crd, Lig_prmtop, Lig_crd)

    
    print("displacement vector (x,y,z)=", Native_displacement )
    

    
    #print('Trans vectors are:', '\n'.join([str(lst) for lst in range(10)]))
    
    
