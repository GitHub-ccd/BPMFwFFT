#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:28:53 2019

@author: dminh
"""
import IO
import numpy as np

def _get_molecule_center_of_mass(inpcrd_file_name, prmtop_file_name):
    """
    return the center of mass of self._crd
    """
    _prmtop = dict()
    _prmtop = IO.PrmtopLoad(prmtop_file_name).get_parm_for_grid_calculation()
    center_of_mass = np.zeros([3], dtype=float)
    masses = _prmtop["MASS"]
    _crd = IO.InpcrdLoad(inpcrd_file_name).get_coordinates()
    natoms = _prmtop["POINTERS"]["NATOM"]
    if (_crd.shape[0] != natoms) or (_crd.shape[1] != 3):
        raise RuntimeError("coordinates in %s has wrong shape"%inpcrd_file_name)
    for atom_ind in range(len(_crd)):
        center_of_mass += masses[atom_ind] * _crd[atom_ind]
    total_mass = masses.sum()
    if total_mass == 0:
        raise RuntimeError("zero total mass")
    return center_of_mass / total_mass


if __name__ == "__main__":
    inpcrd_file_name="/home/chamila/test-3/2.redock/1.amber/1A2K_C:AB/ligand.inpcrd"
    prmtop_file_name="/home/chamila/test-3/2.redock/1.amber/1A2K_C:AB/ligand.prmtop"
    Lig_R=_get_molecule_center_of_mass(inpcrd_file_name, prmtop_file_name)
    print("Ligan x, y, z =", Lig_R[0], Lig_R[1], Lig_R[2])
    
    inpcrd_file_name="/home/chamila/test-3/2.redock/1.amber/1A2K_C:AB/receptor.inpcrd"
    prmtop_file_name="/home/chamila/test-3/2.redock/1.amber/1A2K_C:AB/receptor.prmtop"
    Rec_R=_get_molecule_center_of_mass(inpcrd_file_name, prmtop_file_name)
    print("receptor x, y, z  =", Rec_R[0], Rec_R[1], Rec_R[2])
    
    r_magnitude= np.sqrt(((Lig_R[0]-Rec_R[0])**2)+((Lig_R[1]-Rec_R[1])**2)+((Lig_R[2]-Rec_R[2])**2))
    print("displacement magnitude  =", r_magnitude)
    print("displacement direction  unit(x,y,z)=", ((Lig_R[0]-Rec_R[0])/r_magnitude), ((Lig_R[1]-Rec_R[1])/r_magnitude), ((Lig_R[2]-Rec_R[2])/r_magnitude))
    
    