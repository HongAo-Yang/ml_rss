import ase.io
import os
from subprocess import run
from quippy import descriptors
import numpy as np
from scipy.sparse.linalg import LinearOperator, svds


class RSS:
    def __init__(self,
                 atomic_num=None,
                 dimer_box_min_max=None):
        self.atomic_num = atomic_num
        self.dimer_box_min_max = dimer_box_min_max

    def scale_dimer(self, n_steps):
        box_len = self.dimer_box_min_max[0]
        dimer_min = self.dimer_box_min_max[1]
        dimer_max = self.dimer_box_min_max[2]

        with open("scaled_dimer.extxyz", "w") as f:
            for Zi in self.atomic_num:
                for Zj in self.atomic_num:
                    if Zi < Zj:
                        continue

                    dimer = ase.Atoms(numbers=[Zi, Zj],
                                      cell=[[box_len, 0.0, 0.0],
                                            [0.0, box_len, 0.0],
                                            [0.0, 0.0, box_len]],
                                      pbc=[True, True, True])
                    dimer.info['config_type'] = 'dimer'
                    dimer.info['gap_rss_nonperiodic'] = True

                    for s_i in range(n_steps + 1):
                        s = dimer_min+(dimer_max-dimer_min) * \
                            float(s_i)/float(n_steps)
                        dimer.set_positions([[0.0, 0.0, 0.0],
                                             [s, 0.0, 0.0]])
                        ase.io.write(f, dimer, format="extxyz")

    @staticmethod
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

    @staticmethod
    def create_airss(input_file_name,
                     struct_number,
                     output_file_name,
                     remove_tmp_files=False):
        command = "./buildcell"
        unique_starting_index = 0
        output_file = open(output_file_name, 'w')
        log_file = open("create_airss.log", 'w')
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

            ase.io.write(output_file, at,
                         format="extxyz")
            unique_starting_index += 1
            if remove_tmp_files:
                os.remove(tmp_file_name)

    @staticmethod
    def calculate_descriptor_vec(input_file_name,
                                 selection_descriptor,
                                 output_file_name):
        ats = ase.io.read(input_file_name, ':')
        desc = descriptors.Descriptor(selection_descriptor+" average")
        for at in ats:
            at.info["descriptor_vec"] = desc.calc(at)['data']
        ase.io.write(output_file_name, ats)

    @staticmethod
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

    @staticmethod
    def select_by_descriptor(input_file_name,
                             random_struct_num,
                             selection_method,
                             method_kwargs,
                             output_file_name):
        ats = ase.io.read(input_file_name, ':')
        if selection_method == "CUR":
            selected_ats = RSS.select_by_descriptor_CUR(ats=ats,
                                                        random_struct_num=random_struct_num,
                                                        kernel_exp=method_kwargs['kernel_exp']
                                                        )
        else:
            selected_ats = ats

        for at in selected_ats:
            del at.info["descriptor_vec"]
        ase.io.write(output_file_name, selected_ats)


if __name__ == "__main__":
    dimer_n_steps = 40
    rss = RSS(atomic_num=[31, 8],
              dimer_box_min_max=[20.0, 1.0, 5.0])
    rss.scale_dimer(dimer_n_steps)

    RSS.create_airss(input_file_name='Ga2O3.cell',
                     struct_number=10,
                     output_file_name='random_structs.group_0.extxyz',
                     remove_tmp_files=False)

    RSS.calculate_descriptor_vec(input_file_name='random_structs.group_0.extxyz',
                                 selection_descriptor='soap l_max=12 n_max=12 atom_sigma=0.5 cutoff=10',
                                 output_file_name='descriptor_vec.random_structs.group_0.extxyz')

    RSS.select_by_descriptor(input_file_name='descriptor_vec.random_structs.group_0.extxyz',
                             random_struct_num=2,
                             selection_method="CUR",
                             method_kwargs={'kernel_exp': 4},
                             output_file_name="selected_by_desc.random_structs.group_0.extxyz")

    RSS.VASP_generate_setup_file(input_dir_name="VASP_inputs",
                                 input_file_name="selected_by_desc.random_structs.group_0.extxyz",
                                 output_dir_name="VASP_inputs_selected")
