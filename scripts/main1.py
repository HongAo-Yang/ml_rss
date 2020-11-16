from RSS import *

if __name__ == "__main__":
    my_timer = timer()
    my_timer.start()

    create_airss(input_file_name='GaO_diamond_interface.cell',
                 struct_number=10000,
                 output_file_name='random_structs.group_0.extxyz')
    my_timer.barrier()

    calculate_descriptor_vec(input_file_name='random_structs.group_0.extxyz',
                             selection_descriptor='soap l_max=8 n_max=8 atom_sigma=0.5 cutoff=5.5',
                             output_file_name='descriptor_vec.random_structs.group_0.extxyz')
    my_timer.barrier()

    select_by_descriptor(input_file_name='descriptor_vec.random_structs.group_0.extxyz',
                         random_struct_num=200,
                         selection_method="CUR",
                         method_kwargs={'kernel_exp': 4},
                         output_file_name="selected_by_desc.random_structs.group_0.extxyz")
    my_timer.barrier()

