from RSS import *

if __name__ == "__main__":
    import os
    my_timer = timer()
    my_timer.start()
	
    minimize_structures(input_file_name='random_structs.extxyz',
                        output_file_name='RSS_results',
                        GAP_control='GAP_potential/gap_file.xml',
                        GAP_label='Potential xml_label=GAP_2021_11_1_480_8_4_41_832',
                        scalar_pressure=0,
                        scalar_pressure_exponential_width=0,
                        max_steps=2000,
                        force_tol=0.01,
                        stress_tol=0.05,
                        config_min=0,
                        config_num=1)
    my_timer.barrier()

        

