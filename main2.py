from RSS import *
import os

if __name__ == "__main__":
    num_process = 6
    my_timer = timer()
    my_timer.start()

    create_airss(input_file_name='Ga2O3.cell',
                 struct_number=100,
                 output_file_name='random_structs.extxyz',
                 remove_tmp_files=True,
                 num_process=num_process)
    my_timer.barrier()

    minimize_structures(input_file_name='random_structs.extxyz',
                        output_file_name='RSS_results',
                        GAP_control='GAP_potential/gap_file.xml',
                        GAP_label='Potential xml_label=GAP_2020_10_7_0_15_16_39_727',
                        scalar_pressure=0,
                        scalar_pressure_exponential_width=0,
                        max_steps=4,
                        force_tol=0.01,
                        stress_tol=0.01,
                        config_min=0,
                        config_num=30,
                        num_process=num_process)
    my_timer.barrier()

    os.system("cat RSS_results.out.*_*.extxyz > RSS_results.extxyz")

    # select from RSS results with Boltzmann weighted flat histogram
    select_by_flat_histo(input_file_name='RSS_results.extxyz',
                         minim_select_flat_histo_n=20,
                         kT=0.3,
                         output_file_name='selected_histo.RSS_results.extxyz')

    # select from RSS minima using CUR on descriptor vector
    calculate_descriptor_vec(input_file_name='selected_histo.RSS_results.extxyz',
                             selection_descriptor='soap l_max=12 n_max=12 atom_sigma=0.75 cutoff=10',
                             output_file_name='descriptor_vec.selected_histo.RSS_results.extxyz',
                             num_process=num_process)
    my_timer.barrier()

    select_by_descriptor(input_file_name='descriptor_vec.selected_histo.RSS_results.extxyz',
                         random_struct_num=10,
                         selection_method="CUR",
                         method_kwargs={'kernel_exp': 4},
                         output_file_name="selected_CUR.RSS_results.extxyz")
    my_timer.barrier()

    ## select from trajectories corresponding to selected minima ##########################################################
    # 1. select trajectories
    select_traj_of_minima(infiles=['selected_CUR.RSS_results.extxyz'],
                          outfile='selected_RSS_trajectories.extxyz')   # 'infiles' should be list

    # 2. select from RSS trajectories by Boltzmann-weighted flat histogram
    select_by_flat_histo(input_file_name='selected_RSS_trajectories.extxyz',
                         minim_select_flat_histo_n=20,
                         kT=0.3,
                         output_file_name='selected_histo.selected_RSS_trajectories.extxyz')

    # 3. calculate descriptor vectors for all low symmetry random cells
    calculate_descriptor_vec(input_file_name='selected_histo.selected_RSS_trajectories.extxyz',
                             selection_descriptor='soap l_max=12 n_max=12 atom_sigma=0.75 cutoff=10',
                             output_file_name='descriptor_vec.selected_histo.selected_RSS_trajectories.extxyz',
                             num_process=num_process)
    my_timer.barrier()

    select_by_descriptor(input_file_name='descriptor_vec.selected_histo.selected_RSS_trajectories.extxyz',
                         random_struct_num=10,
                         selection_method="CUR",
                         method_kwargs={'kernel_exp': 4},
                         output_file_name="selected_by_desc.selected_RSS_trajectories.extxyz")
    my_timer.barrier()
