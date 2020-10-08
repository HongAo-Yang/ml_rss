from RSS import *

if __name__ == "__main__":
    num_process = 8
    my_timer = timer()
    my_timer.start()

    create_airss(input_file_name='Ga2O3.cell',
                 struct_number=100,
                 output_file_name='random_structs.group_0.extxyz',
                 num_process=num_process)
    my_timer.barrier()

    LAMMPS_generate_data_file(input_dir_name="LAMMPS_inputs",
                              input_file_name='random_structs.group_0.extxyz',
                              output_dir_name="LAMMPS_inputs_generated",
                              num_process=num_process)
    my_timer.barrier()
