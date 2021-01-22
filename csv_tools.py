import csv
import glob
import os


def get_file_name_and_label_from_csv(csv_path, desired_microscope="Leica"):

    with open(csv_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';', quotechar='|')

        file_paths = []
        labels = []

        for row in csv_reader:
            image_name, class_id, microscope = row
            if microscope == desired_microscope:
                file_path = glob.glob(os.path.join("data","**",image_name))
                if not file_path:
                    continue
                file_paths.append(file_path[0])
                labels.append(int(class_id)-1)
    return file_paths, labels

