import os

def get_data(script_id):
    data_dict = {
        1:'scripts/dataset1.sh',
        2:'scripts/dataset2.sh',
        3:'scripts/dataset3.sh',
        4:'scripts/dataset4.sh'
    }
    os.system(f"sh {data_dict[script_id]}")
