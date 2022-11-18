from RSS import *
import os

if __name__ == "__main__":
    my_timer = timer()
    my_timer.start()

    os.system("cat RSS_results.out.*_*.extxyz > RSS_results.extxyz")

    ## select from trajectories corresponding to selected minima ##########################################################
    # 1. select trajectories
    select_traj_of_minima(infiles=['RSS_results.extxyz'],
                          outfile='selected_RSS_trajectories.extxyz')   # 'infiles' should be list
    my_timer.barrier()

    # # 2. select from RSS trajectories by Boltzmann-weighted flat histogram
    select_by_flat_histo(input_file_name='selected_RSS_trajectories.extxyz',
                         minim_select_flat_histo_n=2500,
                         kT=0.1,
                         output_file_name='selected_histo.selected_RSS_trajectories.extxyz')
    my_timer.barrier()

    # # 3. calculate descriptor vectors for all low symmetry random cells
    calculate_descriptor_vec(input_file_name='selected_histo.selected_RSS_trajectories.extxyz',
                             selection_descriptor='soap l_max=6 n_max=12 atom_sigma=0.5 cutoff=8 add_species=T',
                             output_file_name='descriptor_vec.selected_histo.selected_RSS_trajectories.extxyz')
    my_timer.barrier()

    select_by_descriptor(input_file_name='descriptor_vec.selected_histo.selected_RSS_trajectories.extxyz',
                         random_struct_num=100,
                         selection_method="CUR",
                         method_kwargs={'kernel_exp': 4},
                         output_file_name="selected_by_desc.selected_RSS_trajectories.extxyz")
    my_timer.barrier()
