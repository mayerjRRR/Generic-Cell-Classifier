import csv
import os

def get_file_name_and_label_from_csv(csv_path, desired_microscope="Leica"):

    with open(csv_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';', quotechar='|')

        file_paths_and_labels = []

        for row in csv_reader:
            image_name, class_id, microscope = row
            if microscope == desired_microscope:
                file_paths_and_labels.append([image_name,class_id])
    return file_paths_and_labels

get_file_name_and_label_from_csv(os.path.join("data", "piss.csv"))
