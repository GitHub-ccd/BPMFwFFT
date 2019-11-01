"""
This is to generate interaction energies and corresponding translational vectors, 
given a fixed receptor and an ensemble of ligand coordinates (including rotations and/or configurations)
"""
from __future__ import print_function

import numpy as np
import netCDF4
import sys

import IO
from grids import RecGrid
from grids import LigGrid


KB = 0.001987204134799235


class Sampling(object):
    Grid_displacement = []
    
    def __init__(self, rec_prmtop, lj_sigma_scal_fact, rec_inpcrd, 
                        bsite_file, grid_nc_file,
                        lig_prmtop, lig_inpcrd,
                        lig_coord_ensemble,
                        energy_sample_size_per_ligand,
                        output_nc,
                        temperature=300.):
        """
        :param rec_prmtop: str, name of receptor prmtop file
        :param lj_sigma_scal_fact: float, used to check consitency when loading receptor and ligand grids
        :param rec_inpcrd: str, name of receptor inpcrd file
        :param bsite_file: None or str, name of file defining the box, the same as
        from AlGDock pipeline. "measured_binding_site.py"
        :param grid_nc_file: str, name of receptor precomputed grid netCDF file
        :param lig_prmtop: str, name of ligand prmtop file
        :param lig_inpcrd: str, name of ligand inpcrd file
        :param lig_coord_ensemble: list of 2d array, each array is an ligand coordinate
        :param energy_sample_size_per_ligand: int, number of energies and translational vectors to store for each ligand crd
        :param output_nc: str, name of nc file
        :param temperature: float
        """
        self._energy_sample_size_per_ligand = energy_sample_size_per_ligand
        self._beta = 1./ temperature / KB

        rec_grid = self._create_rec_grid(rec_prmtop, lj_sigma_scal_fact, rec_inpcrd, 
                                        bsite_file, grid_nc_file)
        self._rec_crd = rec_grid.get_crd()

        self._lig_grid = self._create_lig_grid(lig_prmtop, lj_sigma_scal_fact, lig_inpcrd, rec_grid)

        self._lig_coord_ensemble = self._load_ligand_coor_ensemble(lig_coord_ensemble)

        # This Debug
        #print("\n Rec origin :", rec_grid._grid['origin'])
        #print("\n Rec spacing :", rec_grid._grid['spacing'][0])
        #print("\n Rec upper most corner :", rec_grid._uper_most_corner_crd)
        #print("\n Lig origin :",  rec_grid._origin_crd)
        #print("\n Lig upper most corner :",  rec_grid._uper_most_corner_crd)
        #Calculate native displcaement 
        Lig_R=self.center_of_mass(lig_inpcrd, lig_prmtop)
        Rec_R=self.center_of_mass(rec_inpcrd, rec_prmtop)
        Native_displacement=(Lig_R - Rec_R)
        #print("\n ***Native Displacement vector**** =", Native_displacement)
        grid_center = (rec_grid._origin_crd + rec_grid._uper_most_corner_crd) / 2.
        self.Grid_displacement = Native_displacement/rec_grid._grid['spacing'][0]
        self.Grid_displacement = np.round(self.Grid_displacement)
        self.Grid_displacement = np.round(grid_center+self.Grid_displacement)

        self._nc_handle = self._initialize_nc(output_nc)

    def center_of_mass(self, inpcrd_file_name, prmtop_file_name):
        """
        return the center of mass of self._crd
        """
        prmtop = dict()
        prmtop = IO.PrmtopLoad(prmtop_file_name).get_parm_for_grid_calculation()
        com = np.zeros([3], dtype=float)
        masses = prmtop["MASS"]
        crd_temp = IO.InpcrdLoad(inpcrd_file_name).get_coordinates()
        natoms = prmtop["POINTERS"]["NATOM"]
        if (crd_temp.shape[0] != natoms) or (crd_temp.shape[1] != 3):
            raise RuntimeError("coordinates in %s has wrong shape"%inpcrd_file_name)
        for atom_ind in range(len(crd_temp)):
            com += masses[atom_ind] * crd_temp[atom_ind]
        total_mass = masses.sum()
        if total_mass == 0:
            raise RuntimeError("zero total mass")
        return com / total_mass

    def _create_rec_grid(self, rec_prmtop, lj_sigma_scal_fact, rec_inpcrd, bsite_file, grid_nc_file):
        rec_grid = RecGrid(rec_prmtop, lj_sigma_scal_fact, rec_inpcrd, bsite_file, 
                            grid_nc_file, new_calculation=False)
        return rec_grid

    def _create_lig_grid(self, lig_prmtop, lj_sigma_scal_fact, lig_inpcrd, rec_grid):
        lig_grid = LigGrid(lig_prmtop, lj_sigma_scal_fact, lig_inpcrd, rec_grid)
        return lig_grid

    def _load_ligand_coor_ensemble(self, lig_coord_ensemble):
        assert len(lig_coord_ensemble.shape) == 3, "lig_coord_ensemble must be 3-D array."
        ensemble = lig_coord_ensemble
        natoms = self._lig_grid.get_natoms()
        
        for i in range(len(ensemble)):
            if (ensemble[i].shape[0] != natoms) or (ensemble[i].shape[1] != 3):
                raise RuntimeError("Ligand crd %d does not have correct shape"%i)
        return ensemble

    def _initialize_nc(self, output_nc):
        nc_handle = netCDF4.Dataset(output_nc, mode="w", format="NETCDF4")

        nc_handle.createDimension("three", 3)
        rec_natoms = self._rec_crd.shape[0]
        nc_handle.createDimension("rec_natoms", rec_natoms)

        lig_natoms = self._lig_grid.get_natoms()
        nc_handle.createDimension("lig_natoms", lig_natoms)
        nc_handle.createDimension("lig_sample_size", self._lig_coord_ensemble.shape[0])

        nc_handle.createDimension("energy_sample_size_per_ligand", self._energy_sample_size_per_ligand)


        nc_handle.createVariable("rec_positions", "f8", ("rec_natoms", "three"))
        nc_handle.variables["rec_positions"][:,:] = self._rec_crd

        nc_handle.createVariable("lig_positions", "f8", ("lig_sample_size", "lig_natoms", "three"))
        nc_handle.createVariable("lig_com", "f8", ("lig_sample_size", "three"))
        nc_handle.createVariable("volume", "f8", ("lig_sample_size"))
        nc_handle.createVariable("nr_grid_points", "i8", ("lig_sample_size"))

        nc_handle.createVariable("exponential_sums", "f8", ("lig_sample_size"))
        nc_handle.createVariable("log_of_divisors",  "f8", ("lig_sample_size"))

        nc_handle.createVariable("mean_energy",  "f8", ("lig_sample_size"))
        nc_handle.createVariable("min_energy",  "f8", ("lig_sample_size"))
        nc_handle.createVariable("energy_std",  "f8", ("lig_sample_size"))

        nc_handle.createVariable("resampled_energies", "f8", ("lig_sample_size", "energy_sample_size_per_ligand"))
        nc_handle.createVariable("resampled_trans_vectors", "i8", ("lig_sample_size", "energy_sample_size_per_ligand", "three"))

        nc_handle = self._write_grid_info(nc_handle)
        return nc_handle

    def _write_grid_info(self, nc_handle):
        """
        write grid info, "x", "y", "z" ...
        """
        data = self._lig_grid.get_grids()
        grid_func_names = self._lig_grid.get_grid_func_names()
        keys = [key for key in data.keys() if key not in grid_func_names]

        for key in keys:
            for dim in data[key].shape:
                dim_name = "%d"%dim
                if dim_name not in nc_handle.dimensions.keys():
                    nc_handle.createDimension(dim_name, dim)

        for key in keys:
            if data[key].dtype == int:
                store_format = "i8"
            elif data[key].dtype == float:
                store_format = "f8"
            else:
                raise RuntimeError( "Unsupported dtype %s"%data[key].dtype )
            dimensions = tuple([ "%d"%dim for dim in data[key].shape ])
            nc_handle.createVariable(key, store_format, dimensions)

        for key in keys:
            nc_handle.variables[key][:] = data[key]
        return nc_handle

    def _save_data_to_nc(self, step):
        self._nc_handle.variables["lig_positions"][step, :, :] = self._lig_grid.get_crd()

        self._nc_handle.variables["lig_com"][step, :] = self._lig_grid.get_initial_com()

        self._nc_handle.variables["volume"][step] = self._lig_grid.get_box_volume()

        self._nc_handle.variables["nr_grid_points"][step] = self._lig_grid.get_number_translations()

        self._nc_handle.variables["exponential_sums"][step] = self._exponential_sum

        self._nc_handle.variables["log_of_divisors"][step] = self._log_of_divisor

        self._nc_handle.variables["mean_energy"][step] = self._mean_energy
        self._nc_handle.variables["min_energy"][step] = self._min_energy
        self._nc_handle.variables["energy_std"][step] = self._energy_std

        self._nc_handle.variables["resampled_energies"][step,:] = self._resampled_energies

        self._nc_handle.variables["resampled_trans_vectors"][step,:,:] = self._resampled_trans_vectors
        return None

    def _do_fft(self, step):
        print("Doing FFT for step %d"%step)
        lig_conf = self._lig_coord_ensemble[step]
        self._lig_grid.cal_grids(molecular_coord = lig_conf)


        energies = self._lig_grid.get_meaningful_energies()
        self._mean_energy = energies.mean()
        self._min_energy  = energies.min()
        self._energy_std  = energies.std()
        print("Number of finite energy samples", energies.shape[0])

        exp_energies = -self._beta * energies
        self._log_of_divisor = exp_energies.max()
        exp_energies = np.exp(exp_energies - self._log_of_divisor)
        self._exponential_sum = exp_energies.sum()
        exp_energies /= self._exponential_sum
        #sel_ind = np.random.choice(exp_energies.shape[0], size=self._energy_sample_size_per_ligand, p=exp_energies, replace=False)
        del exp_energies

        #self._resampled_energies = [energies[ind] for ind in sel_ind]
        self._lig_grid.set_meaningful_energies_to_none()
        trans_vectors = self._lig_grid.get_meaningful_corners()       
        
        print("******Grid_displacement", self.Grid_displacement)
        
        for i in range(energies.shape[0]):
            if trans_vectors[i][0] == self.Grid_displacement[0] and trans_vectors[i][1] == self.Grid_displacement[1] and trans_vectors[i][2] == self.Grid_displacement[2]:
                print("The Native Pose energy ", energies[i],"\n trans_vectors", trans_vectors[i])


        # This Debug
        print("\n ***energies:\n", energies, "\n length", len(energies))
        sys.exit(print("\n ***trans_vectors:\n", trans_vectors, "\n length", len(trans_vectors)))                    
        

        #self._resampled_trans_vectors = [trans_vectors[ind] for ind in sel_ind]
        del energies
        del trans_vectors

        self._resampled_energies = np.array(self._resampled_energies, dtype=float)
        self._resampled_trans_vectors = np.array(self._resampled_trans_vectors, dtype=int)

        self._save_data_to_nc(step)
        return None

    def run_sampling(self):
        """   
        """
        for step in range(self._lig_coord_ensemble.shape[0]):
            self._do_fft(step)

            print("Min energy", self._min_energy)
            print("Mean energy", self._mean_energy)
            print("STD energy", self._energy_std)
            print("Initial center of mass", self._lig_grid.get_initial_com())
            print("Grid volume", self._lig_grid.get_box_volume())
            print("Number of translations", self._lig_grid.get_number_translations())
            print("-------------------------------\n\n")

        self._nc_handle.close()
        return None

#
#TODO   the class above assumes that the resample size is smaller than number of meaningful energies
#       in general, the number of meaningful energies can be very smaller or even zero (no energy)
#       when the number of meaningful energies is zero, that stratum contributes n_points zeros to the exponential mean
#
#       so when needs to consider separately 3 cases:
#           len(meaningful energies) == 0
#           0< len(meaningful energies) <= resample size
#           len(meaningful energies) > resample size
#


class Sampling_PL(Sampling):
    
    def _write_data_key_2_nc(self, data, key):
        if data.shape[0] == 0:
            return None

        for dim in data.shape:
            dim_name = "%d"%dim
            if dim_name not in self._nc_handle.dimensions.keys():
                self._nc_handle.createDimension(dim_name, dim)

        if data.dtype == int:
            store_format = "i8"
        elif data.dtype == float:
            store_format = "f8"
        else:
            raise RuntimeError("unsupported dtype %s"%data.dtype)
        dimensions = tuple(["%d"%dim for dim in data.shape])
        self._nc_handle.createVariable(key, store_format, dimensions)

        self._nc_handle.variables[key][:] = data
        return None

    def _initialize_nc(self, output_nc):
        """
        """
        nc_handle = netCDF4.Dataset(output_nc, mode="w", format="NETCDF4")

        nc_handle.createDimension("three", 3)
        rec_natoms = self._rec_crd.shape[0]
        nc_handle.createDimension("rec_natoms", rec_natoms)

        lig_natoms = self._lig_grid.get_natoms()
        nc_handle.createDimension("lig_natoms", lig_natoms)
        nc_handle.createDimension("lig_sample_size", self._lig_coord_ensemble.shape[0])

        #nc_handle.createDimension("energy_sample_size_per_ligand", self._energy_sample_size_per_ligand)

        nc_handle.createVariable("rec_positions", "f8", ("rec_natoms", "three"))
        nc_handle.variables["rec_positions"][:,:] = self._rec_crd

        nc_handle.createVariable("lig_positions", "f8", ("lig_sample_size", "lig_natoms", "three"))
        nc_handle.createVariable("lig_com", "f8", ("lig_sample_size", "three"))
        nc_handle.createVariable("volume", "f8", ("lig_sample_size"))
        nc_handle.createVariable("nr_grid_points", "i8", ("lig_sample_size"))
        nc_handle.createVariable("nr_finite_energy", "i8", ("lig_sample_size"))

        nc_handle.createVariable("exponential_sums", "f8", ("lig_sample_size"))
        nc_handle.createVariable("log_of_divisors",  "f8", ("lig_sample_size"))

        nc_handle.createVariable("mean_energy",  "f8", ("lig_sample_size"))
        nc_handle.createVariable("min_energy",  "f8", ("lig_sample_size"))
        nc_handle.createVariable("energy_std",  "f8", ("lig_sample_size"))

        #nc_handle.createVariable("resampled_energies", "f8", ("lig_sample_size", "energy_sample_size_per_ligand"))
        #nc_handle.createVariable("resampled_trans_vectors", "i8", ("lig_sample_size", "energy_sample_size_per_ligand", "three"))

        nc_handle = self._write_grid_info(nc_handle)
        return nc_handle

    def _save_data_to_nc(self, step):
        self._nc_handle.variables["lig_positions"][step, :, :] = self._lig_grid.get_crd()

        self._nc_handle.variables["lig_com"][step, :] = self._lig_grid.get_initial_com()

        self._nc_handle.variables["volume"][step] = self._lig_grid.get_box_volume()

        self._nc_handle.variables["nr_grid_points"][step] = self._lig_grid.get_number_translations()

        self._nc_handle.variables["nr_finite_energy"][step] = self._nr_finite_energy

        self._nc_handle.variables["exponential_sums"][step] = self._exponential_sum

        self._nc_handle.variables["log_of_divisors"][step] = self._log_of_divisor

        self._nc_handle.variables["mean_energy"][step] = self._mean_energy

        self._nc_handle.variables["min_energy"][step] = self._min_energy

        self._nc_handle.variables["energy_std"][step] = self._energy_std

        self._write_data_key_2_nc(self._resampled_energies, "resampled_energies_%d"%step)

        self._write_data_key_2_nc(self._resampled_trans_vectors, "resampled_trans_vectors_%d"%step)
        return None

    def _do_fft(self, step):
        print("Doing FFT for step %d"%step)
        lig_conf = self._lig_coord_ensemble[step]
        self._lig_grid.cal_grids(molecular_coord = lig_conf)

        energies = self._lig_grid.get_meaningful_energies()
        self._nr_finite_energy = energies.shape[0]
        print("Number of finite energy samples", self._nr_finite_energy)

        if energies.shape[0] > 0:

            self._mean_energy = energies.mean()
            self._min_energy  = energies.min()
            self._energy_std  = energies.std()

            exp_energies = -self._beta * energies
            self._log_of_divisor = exp_energies.max()
            exp_energies = np.exp(exp_energies - self._log_of_divisor)
            self._exponential_sum = exp_energies.sum()
            exp_energies /= self._exponential_sum
            
            sample_size = min(exp_energies.shape[0], self._energy_sample_size_per_ligand)
            sel_ind = np.random.choice(exp_energies.shape[0], size=sample_size, p=exp_energies, replace=False)
            del exp_energies

            self._resampled_energies = [energies[ind] for ind in sel_ind]
            del energies
            self._lig_grid.set_meaningful_energies_to_none()

            trans_vectors = self._lig_grid.get_meaningful_corners()
            self._resampled_trans_vectors = [trans_vectors[ind] for ind in sel_ind]
            del trans_vectors

            self._resampled_energies = np.array(self._resampled_energies, dtype=float)
            self._resampled_trans_vectors = np.array(self._resampled_trans_vectors, dtype=int)

        else:

            self._mean_energy = np.inf
            self._min_energy  = np.inf
            self._energy_std  = np.inf

            self._log_of_divisor  = 1.
            self._exponential_sum = 0.

            self._resampled_energies = np.array([], dtype=float)
            del energies
            self._lig_grid.set_meaningful_energies_to_none()

            self._resampled_trans_vectors = np.array([], dtype=float)

        self._save_data_to_nc(step)
        return None


if __name__ == "__main__":
    # test
    rec_prmtop = "../examples/amber/t4_lysozyme/receptor_579.prmtop"
    lj_sigma_scal_fact = 0.8
    rec_inpcrd = "../examples/amber/t4_lysozyme/receptor_579.inpcrd"

    bsite_file = "../examples/amber/t4_lysozyme/measured_binding_site.py"
    grid_nc_file = "../examples/grid/t4_lysozyme/grid.nc"

    lig_prmtop = "../examples/amber/benzene/ligand.prmtop"
    lig_inpcrd = "../examples/amber/benzene/ligand.inpcrd"

    energy_sample_size_per_ligand = 500
    output_nc = "../examples/fft_sampling/t4_benzene/fft_sampling.nc"

    ligand_md_trj_file = "../examples/ligand_md/benzene/trajectory.nc"
    lig_coord_ensemble = netCDF4.Dataset(ligand_md_trj_file, "r").variables["positions"][:]

    sampler = Sampling_PL(rec_prmtop, lj_sigma_scal_fact, rec_inpcrd,
                        bsite_file, grid_nc_file, 
                        lig_prmtop, lig_inpcrd,
                        lig_coord_ensemble,
                        energy_sample_size_per_ligand, 
                        output_nc,
                        temperature=300.)
    sampler.run_sampling()





