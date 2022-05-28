import os
import shutil

import pandas as pd

def setup_dirs_and_preprocess():
    os.system("sh setup.sh")

    def get_all_filePaths(folderPath):
        result = []
        for dirpath, dirnames, filenames in os.walk(folderPath):
            result.extend([os.path.join(dirpath, filename) for filename in filenames if filename[-3:] == 'jpg'])
        return result

    all_paths = []
    all_classes = []
    class_dirs = os.listdir('data/1_extracted/Rock_Dataset/')
    for class_name in class_dirs:
        os.makedirs(os.path.join('data/2_processed', class_name))
        paths_list = get_all_filePaths(f'data/1_extracted/Rock_Dataset/{class_name}')

        for image_path in paths_list:
            source = image_path

            target = os.path.join('data/2_processed', class_name)
            shutil.move(source, target)

            all_paths.append(os.path.join(target, os.path.basename(source)))
            all_classes.append(class_name)

    data = pd.DataFrame(list(zip(all_paths, all_classes)),
                columns =['image_path', 'classes'])
    data.to_csv("training_data.csv")


if __name__ == "__main__":
    setup_dirs_and_preprocess()