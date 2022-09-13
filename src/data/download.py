import os

def get_data(script_id):
    data_dict = {
        1:'src/scripts/dataset1.sh',
        2:'src/scripts/dataset2.sh',
        3:'src/scripts/dataset3.sh',
        4:'src/scripts/dataset4.sh'
    }
    os.system(f"sh {data_dict[script_id]}")
