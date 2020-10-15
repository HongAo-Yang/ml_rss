import time
import ase.io
import os
import sys
import multiprocessing
import numpy as np
import quippy.potential
from matscipy.elasticity import Voigt_6_to_full_3x3_stress
from ase.optimize.precon import PreconLBFGS, Exp
from ase.constraints import UnitCellFilter
from ase.units import GPa
from subprocess import run
from quippy import descriptors
from scipy.sparse.linalg import LinearOperator, svds


class timer:
    def __init__(self) -> None:
        self.start_time = None
        self.last_barrier = None

    def start(self):
        self.start_time = time.time()

    def barrier(self):
        if self.start_time == None:
            self.start()
        time_now = time.time()
        if self.last_barrier == None:
            print('[timer] Cost time %f' % (time_now-self.start_time))
        else:
            print('[timer] Cost time %f' % (time_now-self.last_barrier))
        print('[timer] Total time %f' % (time_now-self.start_time))
        self.last_barrier = time_now


def scale_dimer(atomic_num,
                box_len,
                dimer_min,
                dimer_max,
                dimer_steps,
                output_file_name):
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
                    s = dimer_min+(dimer_max-dimer_min) * \
                        float(s_i)/float(dimer_steps)
                    dimer.set_positions([[0.0, 0.0, 0.0],
                                         [s, 0.0, 0.0]])
                    ase.io.write(f, dimer, format="extxyz")
    print("[log] Finished scale_dimer")


def _VASP_generate_setup_file_single(args):
    '''
    args: dict{'input_dir_name', 'output_dir_name', 'i', 'at'}
    this method should only be called by VASP_generate_setup_file
    '''
    input_dir_name = args['input_dir_name']
    output_dir_name = args['output_dir_name']
    i = args['i']
    at = args['at']
    config_dir_name = os.path.join(output_dir_name, "config_%d" % (i))
    if not os.path.isdir(config_dir_name):
        os.mkdir(config_dir_name)
    cell = at.get_cell()
    if np.dot(np.cross(cell[0, :], cell[1, :]), cell[2, :]) < 0.0:
        t = cell[0, :].copy()
        cell[0, :] = cell[1, :]
        cell[1, :] = t
        at.set_cell(cell, False)
    ase.io.write(os.path.join(config_dir_name, "POSCAR"),
                 at, format="vasp", vasp5=True, sort=True)
    sorted_at = ase.io.read(os.path.join(config_dir_name, "POSCAR"))
    p = at.get_positions()
    sorted_p = sorted_at.get_positions()
    order = []
    for j in range(len(at)):
        order.append(np.argmin([np.sum((x-p[j])**2)
                                for x in sorted_p]))
    with open(os.path.join(config_dir_name, "ASE_VASP_ORDER"), "w") as forder:
        forder.writelines([str(x)+"\n" for x in order])
    os.system("cp {}/* {}".format(input_dir_name, config_dir_name))


def VASP_generate_setup_file(input_dir_name,
                             input_file_name,
                             output_dir_name,
                             num_process=1):
    print("[log] Running VASP_generate_setup_file")
    if not os.path.isdir(input_dir_name):
        raise RuntimeError('input dir %s does not exist' %
                           (input_dir_name))
    if not os.path.isdir(output_dir_name):
        os.mkdir(output_dir_name)
    ats = ase.io.read(input_file_name, ":")
    args = [{'input_dir_name': input_dir_name,
             'output_dir_name': output_dir_name,
             'i': i,
             'at': at}
            for (i, at) in enumerate(ats)]
    pool = multiprocessing.Pool(num_process)
    pool.map(_VASP_generate_setup_file_single, args)
    print("[log] Finished VASP_generate_setup_file")


def _LAMMPS_generate_data_file_single(args):
    '''
    args: dict{'input_dir_name', 'output_dir_name', 'i', 'at'}
    this method should only be called by LAMMPS_generate_data_file
    '''
    input_dir_name = args['input_dir_name']
    output_dir_name = args['output_dir_name']
    i = args['i']
    at = args['at']
    config_dir_name = os.path.join(output_dir_name, "config_%d" % (i))
    if not os.path.isdir(config_dir_name):
        os.mkdir(config_dir_name)
    cell = at.get_cell()
    if np.dot(np.cross(cell[0, :], cell[1, :]), cell[2, :]) < 0.0:
        t = cell[0, :].copy()
        cell[0, :] = cell[1, :]
        cell[1, :] = t
        at.set_cell(cell, False)
    ase.io.write(os.path.join(config_dir_name, "pos.data"),
                 at, format="lammps-data")
    os.system("cp {}/* {}".format(input_dir_name, config_dir_name))


def LAMMPS_generate_data_file(input_dir_name,
                              input_file_name,
                              output_dir_name,
                              num_process=1):
    print("[log] Running LAMMPS_generate_data_file")
    if not os.path.isdir(input_dir_name):
        raise RuntimeError('input dir %s does not exist' %
                           (input_dir_name))
    if not os.path.isdir(output_dir_name):
        os.mkdir(output_dir_name)
    ats = ase.io.read(input_file_name, ":")
    args = [{'input_dir_name': input_dir_name,
             'output_dir_name': output_dir_name,
             'i': i,
             'at': at}
            for (i, at) in enumerate(ats)]
    pool = multiprocessing.Pool(num_process)
    pool.map(_LAMMPS_generate_data_file_single, args)
    print("[log] Finished LAMMPS_generate_data_file")


def _create_airss_single(args):
    '''
    args: dict{'i', 'input_file_name', 'remove_tmp_files'}
    this method should only be called by create_airss
    '''
    i = args['i']
    input_file_name = args['input_file_name']
    remove_tmp_files = args['remove_tmp_files']
    tmp_file_name = "tmp."+str(i)+'.'+input_file_name
    run("./buildcell",
        stdin=open(input_file_name, "r"),
        stdout=open(tmp_file_name, "w"),
        timeout=10.,
        shell=True).check_returncode()
    at = ase.io.read(tmp_file_name)
    at.info["config_type"] = "initial"
    at.info["unique_starting_index"] = i
    if "castep_labels" in at.arrays:
        del at.arrays["castep_labels"]
    if "initial_magmoms" in at.arrays:
        del at.arrays["initial_magmoms"]
    if remove_tmp_files:
        os.remove(tmp_file_name)
    return at


def create_airss(input_file_name,
                 struct_number,
                 output_file_name,
                 num_process=1,
                 remove_tmp_files=False,
                 ):
    print("[log] Running create_airss")
    sys.stdout = open('create_airss.log', 'w')
    sys.stderr = open('create_airss.err', 'w')
    output_file = open(output_file_name, 'w')
    pool = multiprocessing.Pool(num_process)
    args = [{'i': i,
             'input_file_name': input_file_name,
             'remove_tmp_files': remove_tmp_files}
            for i in range(struct_number)]
    ats = pool.map(_create_airss_single, args)
    ase.io.write(output_file, ats,
                 format="extxyz")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print("[log] Finished create_airss")


def _calculate_descriptor_vec_single(args):
    '''
    args: dict{'selection_descriptor', selection_descriptor}
    this method should only be called by calculate_descriptor_vec
    '''
    selection_descriptor = args['selection_descriptor']
    desc_object = descriptors.Descriptor(selection_descriptor+" average")
    at = args['at']
    return desc_object.calc(at)['data']


def calculate_descriptor_vec(input_file_name,
                             selection_descriptor,
                             output_file_name,
                             num_process=1):
    print("[log] Running calculate_descriptor_vec")
    ats = ase.io.read(input_file_name, ':')
    pool = multiprocessing.Pool(num_process)
    args = [{'selection_descriptor': selection_descriptor, 'at': at}
            for at in ats]
    descs_data = pool.map(_calculate_descriptor_vec_single, args)
    for (i, desc_data) in enumerate(descs_data):
        ats[i].info["descriptor_vec"] = desc_data
    ase.io.write(output_file_name, ats)
    print("[log] Finished calculate_descriptor_vec")


def select_by_descriptor_CUR(ats,
                             random_struct_num,
                             stochastic=True,
                             kernel_exp=0.0):
    at_descs = np.array([at.info["descriptor_vec"] for at in ats]).T
    if kernel_exp > 0.0:
        m = np.matmul((np.squeeze(at_descs)).T,
                      np.squeeze(at_descs))**kernel_exp
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
        m, min(max(1, int(random_struct_num/2)), min(m.shape)-1))
    c_scores = np.sum(vt**2, axis=0)/vt.shape[0]
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
    print("[log] Running select_by_descriptor")
    ats = ase.io.read(input_file_name, ':')
    if selection_method == "CUR":
        selected_ats = select_by_descriptor_CUR(ats=ats,
                                                random_struct_num=random_struct_num,
                                                kernel_exp=method_kwargs['kernel_exp']
                                                )
    else:
        selected_ats = ats
    for at in selected_ats:
        del at.info["descriptor_vec"]
    ase.io.write(output_file_name, selected_ats)
    print("[log] Finished select_by_descriptor")


def _minimize_structures_single(args):
    atom_i = args['atom_i']
    atom = args['atom']
    GAP_control = args['GAP_control']
    GAP_label = args['GAP_label']
    max_steps = args['max_steps']
    force_tol = args['force_tol']
    stress_tol = args['stress_tol']
    write_traj = args['write_traj']
    output_file_name = args['output_file_name']
    scalar_pressure = args['scalar_pressure']
    scalar_pressure_exponential_width = args['scalar_pressure_exponential_width']
    log_file_name = output_file_name+'_'+str(atom_i)+'.log'
    log_file = open(log_file_name, 'w')
    sys.stdout = log_file
    sys.stderr = log_file

    calculator = quippy.potential.Potential(args_str=GAP_label,
                                            param_filename=GAP_control)
    calculator.set_default_properties(['energy', 'free_energy', 'forces'])

    atom.set_calculator(calculator)
    scalar_pressure_tmp = scalar_pressure * GPa
    if scalar_pressure_exponential_width > 0.0:
        scalar_pressure_tmp *= np.random.exponential(
            scalar_pressure_exponential_width)
    atom.info["RSS_applied_pressure"] = scalar_pressure_tmp/GPa
    atom = UnitCellFilter(atom, scalar_pressure=scalar_pressure_tmp)
    optimizer = PreconLBFGS(atom, precon=Exp(3), use_armijo=True)
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
        traj_file_name = output_file_name+'_traj_'+str(atom_i)+'.extxyz'
        ase.io.write(traj_file_name, traj)
    del traj[-1].info["minim_stat"]
    traj[-1].info["config_type"] = minim_stat+"_minimum"
    return traj[-1]


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
                        num_process=1):
    atoms = ase.io.read(input_file_name, ":")

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
        config_max = min(config_min+config_num, len(atoms))
    ###
    pool = multiprocessing.Pool(num_process)
    args = [{'atom_i': atom_i,
             'atom': atom,
             'GAP_control': GAP_control,
             'GAP_label': GAP_label,
             'max_steps': max_steps,
             'force_tol': force_tol,
             'stress_tol': stress_tol,
             'write_traj': write_traj,
             'output_file_name': output_file_name,
             'scalar_pressure': scalar_pressure,
             'scalar_pressure_exponential_width': scalar_pressure_exponential_width,
             }
            for (atom_i, atom) in enumerate(atoms[config_min:config_max])]

    minima = pool.map(_minimize_structures_single, args)
    output_file_name_full = output_file_name+'_' + \
        str(config_min)+'_'+str(config_max)+'.extxyz'
    ase.io.write(output_file_name_full, minima)
