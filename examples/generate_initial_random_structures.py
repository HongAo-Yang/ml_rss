from RSS import *

if __name__ == "__main__":
    my_timer = timer()
    my_timer.start()

    create_airss(input_file_name='Ga2O3.cell',
                 struct_number=10000,
                 output_file_name='random_structs.group_0.extxyz',
                 remove_tmp_files=True)
    my_timer.barrier()

    calculate_descriptor_vec(input_file_name='random_structs.group_0.extxyz',
                             selection_descriptor='soap l_max=6 n_max=12 atom_sigma=0.5 cutoff=5 add_species=T',
                             output_file_name='descriptor_vec.random_structs.group_0.extxyz')
    my_timer.barrier()

    select_by_descriptor(input_file_name='descriptor_vec.random_structs.group_0.extxyz',
                         random_struct_num=100,
                         selection_method="CUR",
                         method_kwargs={'kernel_exp': 4},
                         output_file_name="random_structs.extxyz")
    my_timer.barrier()

