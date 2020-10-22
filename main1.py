from RSS import *

if __name__ == "__main__":
    num_process = 8
    my_timer = timer()
    my_timer.start()

    scale_dimer(atomic_num=[31, 8],
                box_len=20,
                dimer_min=1,
                dimer_max=5,
                dimer_steps=40,
                output_file_name="scaled_dimer.extxyz"
                )
    my_timer.barrier()

    create_airss(input_file_name='Ga2O3.cell',
                 struct_number=100,
                 output_file_name='random_structs.group_0.extxyz')
    my_timer.barrier()

    calculate_descriptor_vec(input_file_name='random_structs.group_0.extxyz',
                             selection_descriptor='soap l_max=12 n_max=12 atom_sigma=0.5 cutoff=10',
                             output_file_name='descriptor_vec.random_structs.group_0.extxyz')
    my_timer.barrier()

    select_by_descriptor(input_file_name='descriptor_vec.random_structs.group_0.extxyz',
                         random_struct_num=20,
                         selection_method="CUR",
                         method_kwargs={'kernel_exp': 4},
                         output_file_name="selected_by_desc.random_structs.group_0.extxyz")
    my_timer.barrier()

    VASP_generate_setup_file(input_dir_name="VASP_inputs",
                             input_file_name="selected_by_desc.random_structs.group_0.extxyz",
                             output_dir_name="VASP_inputs_selected")
    my_timer.barrier()
