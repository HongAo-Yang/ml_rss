from RSS import *

if __name__ == "__main__":
    num_process = 8
    my_timer = timer()
    my_timer.start()

    # create_airss(input_file_name='Ga2O3.cell',
    #              struct_number=100,
    #              output_file_name='random_structs.extxyz',
    #              remove_tmp_files=True,
    #              num_process=num_process)
    # my_timer.barrier()

    minimize_structures(input_file_name='random_structs.extxyz',
                        output_file_name='RSS_results',
                        GAP_control='GAP_potential/gap_file.xml',
                        GAP_label='Potential xml_label=GAP_2020_10_7_0_15_16_39_727',
                        scalar_pressure=0,
                        scalar_pressure_exponential_width=0,
                        max_steps=1000,
                        force_tol=0.01,
                        stress_tol=0.01,
                        config_min=0,
                        config_num=4,
                        num_process=4)
    my_timer.barrier()
