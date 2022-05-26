import cv2
from pytz import timezone

import os
import shutil
import imghdr

import pandas as pd

def setup_and_clean_data():
    os.system("sh setup.sh")

    raw_data_dir = 'data/0_raw/'
    extracted_data_dir = 'data/1_extracted/'
    processed_data_dir = 'data/2_processed/'
    extracted_dataset_dir = extracted_data_dir + "Rock_Dataset/"

    zip_file_name = "igneous-metamorphic-sedimentary-rocks-and-minerals.zip"

    classes = os.listdir(extracted_data_dir+"/Rock_Dataset/")

    # modified below code to reduce redundancy and improve reusability
    for c in classes:
        target = f"data/2_processed/{c}"
        os.makedirs(target, exist_ok=True)
        source = f"data/1_extracted/Rock_Dataset/{c}"
        for root, sub, files in os.walk(source):
            for file_name in files:
                path = os.path.join(root, file_name)
                shutil.move(path, target)

    for c in classes:
        path = f"data/2_processed/{c}"
        i=0
        for file_name in os.listdir(path):
            full_file_path = os.path.join(path, file_name)
            new_name = os.path.join(path, f"{c}_{i}.jpg")
            os.rename(full_file_path, new_name)
            i=i+1

    def check_images( s_dir, ext_list):
        bad_images=[]
        bad_ext=[]
        s_list= os.listdir(s_dir)
        for klass in s_list:
            klass_path=os.path.join (s_dir, klass)
            print ('processing class directory ', klass)
            if os.path.isdir(klass_path):
                file_list=os.listdir(klass_path)
                for f in file_list:
                    f_path=os.path.join (klass_path,f)
                    tip = imghdr.what(f_path)
                    if ext_list.count(tip) == 0:
                        bad_images.append(f_path)
                    if os.path.isfile(f_path):
                        try:
                            img=cv2.imread(f_path)
                            shape=img.shape
                        except:
                            print('file ', f_path, ' is not a valid image file')
                            bad_images.append(f_path)
                    else:
                        print('*** fatal error, you a sub directory ', f, ' in class directory ', klass)
            else:
                print ('*** WARNING*** you have files in ', s_dir, ' it should only contain sub directories')
        return bad_images, bad_ext

    source_dir =r'data/2_processed'
    good_exts=['jpg', 'png', 'jpeg', 'gif', 'bmp' ] # list of acceptable extensions
    bad_file_list, bad_ext_list=check_images(source_dir, good_exts)
    if len(bad_file_list) !=0:
        print('improper image files are listed below')
        for i in range (len(bad_file_list)):
            print (bad_file_list[i])
    else:
        print(' no improper image files were found')

    for i in bad_file_list:
        os.remove(i)
        print("% s has been removed successfully" % i)

    target = []
    images = []
    flat_data=[]
    data_dir='data/2_processed'
    CATEGORIES=['igneous rocks','metamorphic rocks','minerals','sedimentary rocks']

    class_paths = []

    for class_var, category in enumerate(CATEGORIES):
    #   class_var=CATEGORIES.index(category)
    #   print(class_var)
        path = os.path.join(data_dir, category)
        class_paths.append(path)

    image_paths = []
    image_classes = []
    for path in class_paths:
        image_paths.extend(list(map(lambda x: os.path.join(path, x), os.listdir(path))))
    print(image_paths)

    image_data = pd.DataFrame({"image_paths":image_paths})
    image_data['classes'] = image_data['image_paths'].apply(lambda x: x.split('/')[-2])

    image_data.to_csv("training_data.csv")

if __name__ == "__main__":
    setup_and_clean_data()