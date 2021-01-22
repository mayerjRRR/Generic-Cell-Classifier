import csv
import glob
import os


def get_file_name_and_label_from_csv(dataset_directory, csv_file_name, desired_microscopes=["Leica"]):
    with open(os.path.join(dataset_directory, csv_file_name), newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';', quotechar='|')

        file_paths = []
        labels = []

        for row in csv_reader:
            image_name, class_id, microscope = row
            if microscope in desired_microscopes:
                file_path = glob.glob(os.path.join("data", "**", image_name))
                if not file_path:
                    continue
                file_paths.append(file_path[0])
                labels.append(int(class_id) - 1)
    if len(file_paths) == 0:
        raise Exception(f"No data found in {dataset_directory}!")
    return file_paths, labels
