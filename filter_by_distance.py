from RSS import *

if __name__ == "__main__":
    my_timer = timer()
    my_timer.start()

    filter_by_distance(input_file_name="selected_histo.selected_RSS_trajectories.extxyz",
                       output_file_name="filtered.selected_histo.selected_RSS_trajectories.extxyz",
                       chemical_symbols=['O', 'Ga'],
                       distance_matrix=[[1.2, 1.5], [1.5, 2.0]])
    my_timer.barrier()
