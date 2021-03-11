import ase.io
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.use('Agg')


def plot_energy_volume(filename_train, filename_test):
    plt.figure()
    atoms = ase.io.read(filename_train, ':')
    Energy = []
    Volume = []
    for atom in atoms:
        Volume.append(atom.get_volume())
        Energy.append(atom.info['REF_energy'])
    Volume = np.array(Volume)
    Energy = np.array(Energy)
    plt.plot(Energy, Volume, 'b.', alpha=0.5, label='train')

    atoms = ase.io.read(filename_test, ':')
    Energy = []
    Volume = []
    for atom in atoms:
        Volume.append(atom.get_volume())
        Energy.append(atom.info['REF_energy'])
    Volume = np.array(Volume)
    Energy = np.array(Energy)
    plt.plot(Energy, Volume, 'r.', alpha=0.5, label='test')
    plt.xlabel('Energy')
    plt.ylabel('Volume')
    plt.legend()
    plt.savefig("Energy_Volume.jpg", dpi=1200)
    plt.close()


def get_atom_distance(filename):
    atoms = ase.io.read(filename, ':')
    Ga_Ga = []
    Ga_O = []
    O_O = []
    for atom in atoms:
        all_distances = atom.get_all_distances()
        chemical_symbols = atom.get_chemical_symbols()
        Ga_index = [index for index, val in enumerate(
            chemical_symbols) if val == 'Ga']
        O_index = [index for index, val in enumerate(
            chemical_symbols) if val == 'O']
        distances = np.triu(all_distances[Ga_index][:, Ga_index])
        Ga_Ga.append(distances.ravel()[np.flatnonzero(distances)])

        distances = all_distances[Ga_index][:, O_index]
        Ga_O.append(distances.ravel()[np.flatnonzero(distances)])

        distances = np.triu(all_distances[O_index][:, O_index])
        O_O.append(distances.ravel()[np.flatnonzero(distances)])

    Ga_Ga = np.array(Ga_Ga).flatten('F')
    Ga_O = np.array(Ga_O).flatten('F')
    O_O = np.array(O_O).flatten('F')
    return({'Ga_Ga': Ga_Ga, 'Ga_O': Ga_O, 'O_O': O_O})


def plot_atom_distance(train_atom_distance, test_atom_distance):
    train_data = train_atom_distance['Ga_Ga']
    test_data = test_atom_distance['Ga_Ga']
    plt.figure()
    plt.hist(train_data, bins=50,
             edgecolor='b', alpha=0.5,
             density=True, label='train')
    plt.hist(test_data, bins=50,
             edgecolor='r', alpha=0.5,
             density=True, label='test')
    plt.xlabel('Ga-Ga distance')
    plt.ylabel('frequency')
    plt.legend()
    plt.savefig("Ga_Ga_distance_hist.jpg", dpi=1200)
    plt.close()

    train_data = train_atom_distance['Ga_O']
    test_data = test_atom_distance['Ga_O']
    plt.figure()
    plt.hist(train_data, bins=50,
             edgecolor='b', alpha=0.5,
             density=True, label='train')
    plt.hist(test_data, bins=50,
             edgecolor='r', alpha=0.5,
             density=True, label='test')
    plt.xlabel('Ga-O distance')
    plt.ylabel('frequency')
    plt.legend()
    plt.savefig("Ga_O_distance_hist.jpg", dpi=1200)
    plt.close()

    train_data = train_atom_distance['O_O']
    test_data = test_atom_distance['O_O']
    plt.figure()
    plt.hist(train_data, bins=50,
             edgecolor='b', alpha=0.5,
             density=True, label='train')
    plt.hist(test_data, bins=50,
             edgecolor='r', alpha=0.5,
             density=True, label='test')
    plt.xlabel('O-O distance')
    plt.ylabel('frequency')
    plt.legend()
    plt.savefig("O_O_distance_hist.jpg", dpi=1200)
    plt.close()


def filter_by_distance_count(input_file_name,
                             chemical_symbols,
                             distance_matrix):

    atoms = ase.io.read(input_file_name, ":", parallel=False)
    count = 0
    print('########', input_file_name, '########')
    for index, atom in enumerate(atoms):
        flag = False
        all_distances = atom.get_all_distances()
        all_distances = all_distances+np.identity(len(all_distances))*100
        all_chemical_symbols = atom.get_chemical_symbols()
        distances_tolerance = np.zeros_like(all_distances)
        for i, chemical_symbol_i in enumerate(chemical_symbols):
            for j, chemical_symbol_j in enumerate(chemical_symbols):
                if j < i:
                    break
                index1 = [index for index, val in enumerate(
                    all_chemical_symbols) if val == chemical_symbol_i]
                index2 = [index for index, val in enumerate(
                    all_chemical_symbols) if val == chemical_symbol_j]
                for k in index1:
                    distances_tolerance[k, index2] = distance_matrix[i][j]
        for i in range(all_distances.shape[0]):
            for j in range(all_distances.shape[1]):
                if i == j:
                    break
                if all_distances[i, j] < distances_tolerance[i, j]:
                    print('structure %d, atom %d (%s) and atom %d (%s) distance %f, tolerance %f' % (
                        index,
                        i, all_chemical_symbols[i],
                        j, all_chemical_symbols[j],
                        all_distances[i, j],
                        distances_tolerance[i, j]))
                    flag = True
        if flag:
            count += 1
    print(input_file_name, ', %d structures filtered' % (count))


def distance_minimum(input_file_name):
    atoms = ase.io.read(input_file_name, ":", parallel=False)
    print('########', input_file_name, '########')
    for index, atom in enumerate(atoms):
        all_distances = atom.get_all_distances()
        all_distances = all_distances+np.identity(len(all_distances))*100
        all_chemical_symbols = atom.get_chemical_symbols()
        index1 = [index for index, val in enumerate(
            all_chemical_symbols) if val == 'O']
        index2 = [index for index, val in enumerate(
            all_chemical_symbols) if val == 'O']
        print('structure %d, O-O distance minimum %f' %
              (index, all_distances[index1][:, index2].min()))


def filter_by_force(input_file_name,
                    output_file_name,
                    force_toleration):
    atoms = ase.io.read(input_file_name, ':')
    atoms_filtered = []
    for atom in atoms:
        if np.abs(atom.get_forces()).max() < force_toleration:
            atoms_filtered.append(atom)
    print(input_file_name, ',original structure %d, now structure %d' %
          (len(atoms), len(atoms_filtered)))
    ase.io.write(output_file_name, atoms_filtered, format='extxyz')


def filter_by_energy_error(input_file_name, output_file_name):
    atoms = ase.io.read(input_file_name, ':')
    energy_error = []
    for atom in atoms:
        energy = atom.info['energy']
        REF_energy = atom.info['REF_energy']
        Num_atom = atom.get_atomic_numbers().size
        energy_error.append(abs(energy-REF_energy)/Num_atom)
    critical_energy_error = sorted(energy_error, reverse=True)[50]
    atoms_filtered = []
    for (i, atom) in enumerate(atoms):
        if energy_error[i] > critical_energy_error:
            atoms_filtered.append(atom)
    ase.io.write(output_file_name, atoms_filtered, format='extxyz')


def filter_unique(input_file_name, output_file_name):
    atoms = ase.io.read(input_file_name, ':')
    data = []
    for atom in atoms:
        data.append([atom.info['unique_starting_index'],
                     atom.info['REF_energy']])
    data = np.array(data)
    unique_data, index = np.unique(data, return_index=True, axis=0)
    atoms_filtered = [atoms[i] for i in index]
    print('original structure number: %d' % (len(atoms)))
    print('unique structure number: %d' % (len(atoms_filtered)))
    ase.io.write(output_file_name, atoms_filtered, format='extxyz')


if __name__ == "__main__":
    filter_unique('Ga2O3_train.extxyz',
                  'out.extxyz')
