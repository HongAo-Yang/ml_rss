import ase.io
import os
from subprocess import run
from quippy import descriptors
import numpy as np
from scipy.sparse.linalg import LinearOperator, svds


def scale_dimer(atomic_num,
                box_len,
                dimer_min,
                dimer_max,
                dimer_steps,
                output_file_name):
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


def VASP_generate_setup_file(input_dir_name,
                             input_file_name,
                             output_dir_name):
    if not os.path.isdir(input_dir_name):
        raise RuntimeError('input dir %s does not exist' %
                           (input_dir_name))
    if not os.path.isdir(output_dir_name):
        os.mkdir(output_dir_name)
    ats = ase.io.read(input_file_name, ":")
    (fbase, fext) = os.path.splitext(input_file_name)
    for (i, at) in enumerate(ats):
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
        for i in range(len(at)):
            order.append(np.argmin([np.sum((x-p[i])**2)
                                    for x in sorted_p]))
        with open(os.path.join(config_dir_name, "ASE_VASP_ORDER"), "w") as forder:
            forder.writelines([str(x)+"\n" for x in order])
        os.system("cp {}/* {}".format(input_dir_name, config_dir_name))


def create_airss(input_file_name,
                 struct_number,
                 output_file_name,
                 remove_tmp_files=False):
    command = "./buildcell"
    unique_starting_index = 0
    output_file = open(output_file_name, 'w')
    log_file = open("create_airss.log", 'w')
    ats = []
    for i in range(struct_number):
        tmp_file_name = "tmp."+str(i)+'.'+input_file_name
        log_file.flush()
        run(command,
            stdin=open(input_file_name),
            stdout=open(tmp_file_name, "w"),
            stderr=log_file,
            timeout=10.,
            shell=True).check_returncode()
        log_file.flush()
        at = ase.io.read(tmp_file_name)
        at.info["config_type"] = "initial"
        at.info["unique_starting_index"] = unique_starting_index
        if "castep_labels" in at.arrays:
            del at.arrays["castep_labels"]
        if "initial_magmoms" in at.arrays:
            del at.arrays["initial_magmoms"]
        ats.append(at)
        unique_starting_index += 1
        if remove_tmp_files:
            os.remove(tmp_file_name)
    ase.io.write(output_file, ats,
                 format="extxyz")


def calculate_descriptor_vec(input_file_name,
                             selection_descriptor,
                             output_file_name):
    ats = ase.io.read(input_file_name, ':')
    desc = descriptors.Descriptor(selection_descriptor+" average")
    for at in ats:
        at.info["descriptor_vec"] = desc.calc(at)['data']
    ase.io.write(output_file_name, ats)


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


if __name__ == "__main__":
    scale_dimer(atomic_num=[31, 8],
                box_len=20,
                dimer_min=1,
                dimer_max=5,
                dimer_steps=40,
                output_file_name="scaled_dimer.extxyz"
                )

    create_airss(input_file_name='Ga2O3.cell',
                 struct_number=100,
                 output_file_name='random_structs.group_0.extxyz',
                 remove_tmp_files=False)

    calculate_descriptor_vec(input_file_name='random_structs.group_0.extxyz',
                             selection_descriptor='soap l_max=12 n_max=12 atom_sigma=0.5 cutoff=10',
                             output_file_name='descriptor_vec.random_structs.group_0.extxyz')

    select_by_descriptor(input_file_name='descriptor_vec.random_structs.group_0.extxyz',
                         random_struct_num=20,
                         selection_method="CUR",
                         method_kwargs={'kernel_exp': 4},
                         output_file_name="selected_by_desc.random_structs.group_0.extxyz")

    VASP_generate_setup_file(input_dir_name="VASP_inputs",
                             input_file_name="selected_by_desc.random_structs.group_0.extxyz",
                             output_dir_name="VASP_inputs_selected")
