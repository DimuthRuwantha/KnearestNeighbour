import os
import csv


def file_reader(split_ratio=0.75):
    """Read all the files in the given folder and load data with 2 data sets as training data and testing data"""
    root_path = 'data/'
    # top_view file_names
    file_names = [file_name for file_name in os.listdir(root_path)]
    file_paths = [root_path + file_path for file_path in file_names]

    # Fault types
    class_titles = [file_name[:-4] for file_name in file_names]

    for file_path in file_paths:

        training, testing = load_data_set(file_path, split_ratio)

    return training, testing


def load_data_set(filename, split, training_set=[], testing_set=[]):

    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        iterator = 0
        for x in range(len(dataset) - 1):
            for y in range(len(dataset[x])-1):
                dataset[x][y] = float(dataset[x][y])
            if iterator < split*lines.line_num:
                training_set.append(dataset[x])
            else:
                testing_set.append(dataset[x])
            iterator += 1
    return training_set, testing_set

if __name__ == "__main__":
    file_reader()
