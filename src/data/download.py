import os


def get_data(dataset_id):
    """_summary_

    Parameters
    ----------
    dataset_id : _type_
        _description_
    """
    data_dict = {
        1: "src/scripts/dataset1.sh",
        2: "src/scripts/dataset2.sh",
        3: "src/scripts/dataset3.sh",
        4: "src/scripts/dataset4.sh",
    }
    if not os.path.exists(os.path.join("data", "1_extracted", f"dataset{dataset_id}")):
        os.system(f"sh {data_dict[dataset_id]}")
