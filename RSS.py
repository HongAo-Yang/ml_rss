import time
import ase.io
import os
import numpy as np
import quippy.potential
import math
from numpy.lib.function_base import append
from matscipy.elasticity import Voigt_6_to_full_3x3_stress
from ase.optimize.precon import PreconLBFGS, Exp
from ase.constraints import UnitCellFilter
from ase.units import GPa
from subprocess import run
from quippy import descriptors
from scipy.sparse.linalg import LinearOperator, svds
from mpi4py import MPI


class timer:
    def __init__(self) -> None:
        self.me = MPI.COMM_WORLD.Get_rank()
        self.start_time = None
        self.last_barrier = None

    def start(self):
        self.start_time = time.time()

    def barrier(self):
        if self.start_time == None:
            self.start()
        time_now = time.time()
        if self.last_barrier == None:
            if self.me == 0:
                print('[timer] Cost time %f' % (time_now - self.start_time))
        else:
            if self.me == 0:
                print('[timer] Cost time %f' % (time_now - self.last_barrier))
        if self.me == 0:
            print('[timer] Total time %f' % (time_now - self.start_time))
        self.last_barrier = time_now


def scale_dimer(atomic_num,
                box_len,
                dimer_min,
                dimer_max,
                dimer_steps,
                output_file_name):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("[log] Running scale_dimer")
        with open(output_file_name, "w") as f:
            for Zi in atomic_num:
                for Zj in atomic_num:
                    if Zi < Zj:
                        continue
                    dimer = ase.Atoms(numbers=[Zi, Zj],
                                      cell=[[box_len, 0.0, 0.0],
                                            [0.0, box_len, 0.0],
                                            [0.0, 0.0, box_len]],
                                      pbc=[True, True, True])
                    dimer.info['config_type'] = 'dimer'
                    dimer.info['gap_rss_nonperiodic'] = True
                    for s_i in range(dimer_steps + 1):
                        s = dimer_min + (dimer_max - dimer_min) * \
                            float(s_i) / float(dimer_steps)
                        dimer.set_positions([[0.0, 0.0, 0.0],
                                             [s, 0.0, 0.0]])
                        ase.io.write(f, dimer, format="extxyz", parallel=False)
        print("[log] Finished scale_dimer")


def VASP_generate_setup_file(input_dir_name,
                             input_file_name,
                             output_dir_name):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    atoms_group = None
    if rank == 0:
        print("[log] Running VASP_generate_setup_file")
        if not os.path.isdir(input_dir_name):
            raise RuntimeError('input dir %s does not exist' %
                               (input_dir_name))
        if not os.path.isdir(output_dir_name):
            os.mkdir(output_dir_name)
        atoms = ase.io.read(input_file_name, ":", parallel=False)
        size = comm.Get_size()
        num_atom_local = len(atoms)//size
        atoms_group = []
        for i in range(size):
            if num_atom_local*size+i < len(atoms):
                index_lo = (num_atom_local+1)*i
                index_hi = (num_atom_local+1)*(i+1)
            else:
                index_lo = len(atoms)-(size-i)*num_atom_local
                index_hi = len(atoms)-(size-i-1)*num_atom_local
            atoms_group.append([atom for atom in atoms[index_lo:index_hi]])
    atoms_local = comm.scatter(atoms_group, root=0)
    for atom in atoms_local:
        unique_name = str()
        if 'unique_starting_index' in atom.info:
            unique_name = unique_name+str(atom.info['unique_starting_index'])
        if 'RSS_minim_iter' in atom.info:
            unique_name = unique_name+'_'+str(atom.info['RSS_minim_iter'])
        config_dir_name = os.path.join(
            output_dir_name, "config_%s" % (unique_name))
        if not os.path.isdir(config_dir_name):
            os.mkdir(config_dir_name)
        cell = atom.get_cell()
        if np.dot(np.cross(cell[0, :], cell[1, :]), cell[2, :]) < 0.0:
            t = cell[0, :].copy()
            cell[0, :] = cell[1, :]
            cell[1, :] = t
            atom.set_cell(cell, False)
        ase.io.write(os.path.join(config_dir_name, "POSCAR"),
                     atom, format="vasp", vasp5=True, sort=True)
        sorted_at = ase.io.read(os.path.join(config_dir_name, "POSCAR"))
        p = atom.get_positions()
        sorted_p = sorted_at.get_positions()
        order = []
        for j in range(len(atom)):
            order.append(np.argmin([np.sum((x - p[j]) ** 2)
                                    for x in sorted_p]))
        with open(os.path.join(config_dir_name, "ASE_VASP_ORDER"), "w") as forder:
            forder.writelines([str(x) + "\n" for x in order])
        os.system("cp {}/* {}".format(input_dir_name, config_dir_name))
    if rank == 0:
        print("[log] Finished VASP_generate_setup_file")


def LAMMPS_generate_setup_file(input_dir_name,
                               input_file_name,
                               output_dir_name):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    atoms_group = None
    if rank == 0:
        print("[log] Running LAMMPS_generate_setup_file")
        if not os.path.isdir(input_dir_name):
            raise RuntimeError('input dir %s does not exist' %
                               (input_dir_name))
        if not os.path.isdir(output_dir_name):
            os.mkdir(output_dir_name)
        atoms = ase.io.read(input_file_name, ":", parallel=False)
        size = comm.Get_size()
        num_atom_local = len(atoms)//size
        atoms_group = []
        for i in range(size):
            if num_atom_local*size+i < len(atoms):
                index_lo = (num_atom_local+1)*i
                index_hi = (num_atom_local+1)*(i+1)
            else:
                index_lo = len(atoms)-(size-i)*num_atom_local
                index_hi = len(atoms)-(size-i-1)*num_atom_local
            atoms_group.append([atom for atom in atoms[index_lo:index_hi]])
    atoms_local = comm.scatter(atoms_group, root=0)
    for atom in atoms_local:
        unique_starting_index = atom.info['unique_starting_index']
        config_dir_name = os.path.join(
            output_dir_name, "config_%d" % (unique_starting_index))
        if not os.path.isdir(config_dir_name):
            os.mkdir(config_dir_name)
        cell = atom.get_cell()
        if np.dot(np.cross(cell[0, :], cell[1, :]), cell[2, :]) < 0.0:
            t = cell[0, :].copy()
            cell[0, :] = cell[1, :]
            cell[1, :] = t
            atom.set_cell(cell, False)
        ase.io.write(os.path.join(config_dir_name, "pos.data"),
                     atom, format="lammps-data")
        os.system("cp {}/* {}".format(input_dir_name, config_dir_name))
    if rank == 0:
        print("[log] Finished LAMMPS_generate_setup_file")


def create_airss(input_file_name,
                 struct_number,
                 output_file_name,
                 remove_tmp_files=False,
                 ):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        print("[log] Running create_airss")
    num_atom_local = struct_number//size
    if num_atom_local*size+rank < struct_number:
        index_lo = (num_atom_local+1)*rank
        index_hi = (num_atom_local+1)*(rank+1)
    else:
        index_lo = struct_number-(size-rank)*num_atom_local
        index_hi = struct_number-(size-rank-1)*num_atom_local
    atoms_local = []
    for i in range(index_lo, index_hi):
        tmp_file_name = "tmp." + str(i) + '.' + input_file_name
        run("./buildcell",
            stdin=open(input_file_name, "r"),
            stdout=open(tmp_file_name, "w"),
            shell=True).check_returncode()
        atom = ase.io.read(tmp_file_name, parallel=False)
        atom.info["config_type"] = "initial"
        atom.info["unique_starting_index"] = i
        if "castep_labels" in atom.arrays:
            del atom.arrays["castep_labels"]
        if "initial_magmoms" in atom.arrays:
            del atom.arrays["initial_magmoms"]
        if remove_tmp_files:
            os.remove(tmp_file_name)
        atoms_local.append(atom)
    comm.barrier()
    atoms_group = comm.gather(atoms_local, root=0)
    if rank == 0:
        atoms = []
        for i in atoms_group:
            for j in i:
                atoms.append(j)
        output_file = open(output_file_name, 'w')
        ase.io.write(output_file,
                     atoms,
                     parallel=False,
                     format="extxyz")
        print("[log] Finished create_airss")
    comm.barrier()


def calculate_descriptor_vec(input_file_name,
                             selection_descriptor,
                             output_file_name):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    atoms_group = None
    if rank == 0:
        print("[log] Running calculate_descriptor_vec")
        atoms = ase.io.read(input_file_name, ':', parallel=False)
        size = comm.Get_size()
        num_atom_local = len(atoms)//size
        atoms_group = []
        for i in range(size):
            if num_atom_local*size+i < len(atoms):
                index_lo = (num_atom_local+1)*i
                index_hi = (num_atom_local+1)*(i+1)
            else:
                index_lo = len(atoms)-(size-i)*num_atom_local
                index_hi = len(atoms)-(size-i-1)*num_atom_local
            atoms_group.append([atom for atom in atoms[index_lo:index_hi]])
    atoms_local = comm.scatter(atoms_group, root=0)
    desc_object = descriptors.Descriptor(selection_descriptor + " average")
    for atom in atoms_local:
        atom.info["descriptor_vec"] = desc_object.calc(atom)['data']
    atoms_group = comm.gather(atoms_local, root=0)
    if rank == 0:
        atoms = []
        for i in atoms_group:
            for j in i:
                atoms.append(j)
        ase.io.write(output_file_name, atoms, parallel=False)
        print("[log] Finished calculate_descriptor_vec")


def select_by_descriptor_CUR(ats,
                             random_struct_num,
                             stochastic=True,
                             kernel_exp=0.0):
    at_descs = np.array([at.info["descriptor_vec"] for at in ats]).T
    if kernel_exp > 0.0:
        m = np.matmul((np.squeeze(at_descs)).T,
                      np.squeeze(at_descs)) ** kernel_exp
    else:
        m = at_descs

    def descriptor_svd(at_descs, num, do_vectors='vh'):
        def mv(v):
            return np.dot(at_descs, v)

        def rmv(v):
            return np.dot(at_descs.T, v)

        A = LinearOperator(at_descs.shape, matvec=mv,
                           rmatvec=rmv, matmat=mv)
        return svds(A, k=num, return_singular_vectors=do_vectors)

    (_, _, vt) = descriptor_svd(
        m, min(max(1, int(random_struct_num / 2)), min(m.shape) - 1))
    c_scores = np.sum(vt ** 2, axis=0) / vt.shape[0]
    if stochastic:
        selected = sorted(np.random.choice(
            range(len(ats)), size=random_struct_num, replace=False, p=c_scores))
    else:
        selected = sorted(np.argsort(c_scores)[-random_struct_num:])
    return [ats[i] for i in selected]


def select_by_descriptor(input_file_name,
                         random_struct_num,
                         selection_method,
                         method_kwargs,
                         output_file_name):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("[log] Running select_by_descriptor")
        ats = ase.io.read(input_file_name, ':', parallel=False)
        if selection_method == "CUR":
            selected_ats = select_by_descriptor_CUR(ats=ats,
                                                    random_struct_num=random_struct_num,
                                                    kernel_exp=method_kwargs['kernel_exp']
                                                    )
        else:
            selected_ats = ats
        for at in selected_ats:
            del at.info["descriptor_vec"]
        ase.io.write(output_file_name, selected_ats, parallel=False)
        print("[log] Finished select_by_descriptor")


def minimize_structures(input_file_name,
                        output_file_name,
                        GAP_control,
                        GAP_label,
                        # applied scalar pressure prefactor
                        scalar_pressure=0,
                        # applied scalar pressure exponential distribution width
                        scalar_pressure_exponential_width=0,
                        max_steps=2000,
                        force_tol=1.0e-3,
                        stress_tol=1.0e-3,
                        config_min=None,
                        config_num=None,
                        write_traj=True,
                        ):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    atoms_group = None
    config_max = None
    if rank == 0:
        if not os.path.isdir("RSS_tmp"):
            os.makedirs("RSS_tmp")
        atoms = ase.io.read(input_file_name, ":", parallel=False)
        # 如果没有指定，则默认最小化所有结构
        if config_min is None:
            config_min = 0
        elif config_min < 0:
            raise RuntimeError("[ERROR] config_min < 0")
        elif config_min >= len(atoms):
            raise RuntimeError("[ERROR] config_min > number of structures")
        if config_num is None:
            config_max = len(atoms)
        else:
            config_max = min(config_min + config_num, len(atoms))
        ###
        atoms = atoms[config_min:config_max]
        size = comm.Get_size()
        num_atom_local = len(atoms)//size
        atoms_group = []
        for i in range(size):
            if num_atom_local*size+i < len(atoms):
                index_lo = (num_atom_local+1)*i
                index_hi = (num_atom_local+1)*(i+1)
            else:
                index_lo = len(atoms)-(size-i)*num_atom_local
                index_hi = len(atoms)-(size-i-1)*num_atom_local
            atoms_group.append([atom for atom in atoms[index_lo:index_hi]])
    atoms_local = comm.scatter(atoms_group, root=0)
    calculator = quippy.potential.Potential(args_str=GAP_label,
                                            param_filename=GAP_control)
    calculator.set_default_properties(['energy', 'free_energy', 'forces'])
    minima_local = []
    for atom in atoms_local:
        unique_starting_index = atom.info['unique_starting_index']
        log_file_name = "RSS_tmp/"+output_file_name + \
            '_' + str(unique_starting_index) + '.log'
        log_file = open(log_file_name, 'w')
        atom.set_calculator(calculator)
        scalar_pressure_tmp = scalar_pressure * GPa
        if scalar_pressure_exponential_width > 0.0:
            scalar_pressure_tmp *= np.random.exponential(
                scalar_pressure_exponential_width)
        atom.info["RSS_applied_pressure"] = scalar_pressure_tmp / GPa
        atom = UnitCellFilter(atom, scalar_pressure=scalar_pressure_tmp)
        optimizer = PreconLBFGS(atom,
                                precon=Exp(3),
                                use_armijo=True,
                                logfile=log_file,
                                master=True)
        traj = []

        def build_traj():
            traj.append(atom.copy())

        optimizer.attach(build_traj)
        optimizer.run(fmax=force_tol, smax=stress_tol, steps=max_steps)
        if optimizer.converged():
            minim_stat = "converged"
        else:
            minim_stat = "unconverged"

        for (traj_at_i, traj_at) in enumerate(traj):
            traj_at.info["RSS_minim_iter"] = traj_at_i
            traj_at.info["config_type"] = "traj"
            traj_at.info["minim_stat"] = minim_stat
            traj_at.info["stress"] = - \
                Voigt_6_to_full_3x3_stress(traj_at.info["stress"])
        if write_traj:
            traj_file_name = "RSS_tmp/" + output_file_name + \
                '_traj_' + str(unique_starting_index) + '.extxyz'
            ase.io.write(traj_file_name, traj, parallel=False)
        del traj[-1].info["minim_stat"]
        traj[-1].info["config_type"] = minim_stat + "_minimum"
        minima_local.append(traj[-1])
    comm.barrier()
    minima_group = comm.gather(minima_local, root=0)
    if rank == 0:
        minima = []
        for i in minima_group:
            for j in i:
                minima.append(j)
        output_file_name_full = output_file_name + '.out.' + \
            str(config_min) + '_' + str(config_max) + '.extxyz'
        ase.io.write(output_file_name_full, minima, parallel=False)


def select_by_flat_histo(input_file_name,
                         minim_select_flat_histo_n,
                         kT,
                         output_file_name):
    if MPI.COMM_WORLD.Get_rank() == 0:
        enthalpies = []
        avail_configs = []
        ats = ase.io.read(input_file_name, ":", parallel=False)
        for at in ats:
            if at.info["config_type"] != "failed_minimum":
                enthalpy = (at.get_potential_energy() + at.get_volume()
                            * at.info["RSS_applied_pressure"] * GPa) / len(at)
                enthalpies.append(enthalpy)
                avail_configs.append(at)
                # compute desired probabilities for flattened histogram
        min_H = np.min(enthalpies)
        config_prob = []
        histo = np.histogram(enthalpies)
        for H in enthalpies:
            bin_i = np.searchsorted(histo[1][1:], H, side='right')
            if bin_i == len(histo[1][1:]):
                bin_i = bin_i - 1
            if histo[0][bin_i] > 0.0:
                p = 1.0 / histo[0][bin_i]
            else:
                p = 0.0
            if kT > 0.0:
                p *= np.exp(-(H - min_H) / kT)
            config_prob.append(p)

        selected_ats = []
        for _ in range(minim_select_flat_histo_n):
            config_prob = np.array(config_prob)
            config_prob /= np.sum(config_prob)
            cumul_prob = np.cumsum(config_prob)  # cumulate prob
            rv = np.random.uniform()
            config_i = np.searchsorted(cumul_prob, rv)
            selected_ats.append(avail_configs[config_i])
            # remove from config_prob by converting to list
            config_prob = list(config_prob)
            del config_prob[config_i]
            # remove from other lists
            del avail_configs[config_i]
            del enthalpies[config_i]
        ase.io.write(output_file_name, selected_ats, parallel=False)


def select_traj_of_minima(outfile, infiles):
    if MPI.COMM_WORLD.Get_rank() == 0:
        with open(outfile, "w") as fout:
            for f in infiles:
                for at in ase.io.read(f, ":", parallel=False):
                    traj = ase.io.read('RSS_tmp/RSS_results_traj_{}.extxyz'.format(
                        at.info['unique_starting_index']), ":", parallel=False)
                    ase.io.write(fout, traj, format="extxyz", parallel=False)


def filter_by_distance(input_file_name,
                       output_file_name,
                       chemical_symbols,
                       distance_matrix):
    '''
    usage example: 
    filter_by_distance("in.extxyz","out.extxyz",['O','Ga'],[[1.5,1.7],[1.7,2.0]])
    '''
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    atoms_group = None
    num_scatter = None
    if rank == 0:
        atoms = ase.io.read(input_file_name, ":", parallel=False)
        print("[log] Running filter_by_distance")
        print("      original struct number: %d" % (len(atoms)))
        size = comm.Get_size()
        max_structure = 30000                     # 每次广播30000个结构
        num_scatter = math.ceil(len(atoms)/max_structure)
        num_atom_local = len(atoms)//size
        atoms_group = [[] for _ in range(num_scatter)]
        for i in range(num_scatter):
            atoms_this_scatter = atoms[i *
                                       max_structure:min((i+1)*max_structure, len(atoms))]
            num_atom_local = len(atoms_this_scatter)//size
            for j in range(size):
                if num_atom_local*size+j < len(atoms_this_scatter):
                    index_lo = (num_atom_local+1)*j
                    index_hi = (num_atom_local+1)*(j+1)
                else:
                    index_lo = len(atoms_this_scatter)-(size-j)*num_atom_local
                    index_hi = len(atoms_this_scatter) - \
                        (size-j-1)*num_atom_local
                atoms_group[i].append(
                    [atom for atom in atoms_this_scatter[index_lo:index_hi]])
    num_scatter = comm.bcast(num_scatter, root=0)
    atoms_local = []
    for i in range(num_scatter):
        tmp = None
        if rank == 0:
            tmp = atoms_group[i]
        atoms_local.append(comm.scatter(tmp, root=0))
    atoms_selected_local = []
    for atoms in atoms_local:
        atoms_selected_single = []
        for atom in atoms:
            selected = True
            all_distances = atom.get_all_distances()
            all_distances = all_distances+np.identity(len(all_distances))*100
            all_chemical_symbols = atom.get_chemical_symbols()
            for i, chemical_symbol_i in enumerate(chemical_symbols):
                for j, chemical_symbol_j in enumerate(chemical_symbols):
                    if j < i:
                        break
                    index1 = [index for index, val in enumerate(
                        all_chemical_symbols) if val == chemical_symbol_i]
                    index2 = [index for index, val in enumerate(
                        all_chemical_symbols) if val == chemical_symbol_j]
                    if all_distances[index1][:, index2].min() < distance_matrix[i][j]:
                        selected = False
                        break
                if not selected:
                    break
            if selected:
                atoms_selected_single.append(atom)
        atoms_selected_local.append(atoms_selected_single)
    comm.barrier()
    atoms_selected_group = []
    for i in range(num_scatter):
        tmp = atoms_selected_local[i]
        atoms_selected_group.append(comm.gather(tmp, root=0))
    if rank == 0:
        atoms_selected = []
        for i in atoms_selected_group:
            for j in i:
                for k in j:
                    atoms_selected.append(k)
        print("      now struct number: %d" % (len(atoms_selected)))
        print("[log] Finished filter_by_distance")
        ase.io.write(output_file_name, atoms_selected, parallel=False)
