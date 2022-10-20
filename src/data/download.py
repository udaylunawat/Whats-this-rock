import os
from utils import timer_func


@timer_func
def get_data(dataset_id: int):
    """Download the dataset with dataset_id.

    Parameters
    ----------
    dataset_id : int
        Dataset number
    """
    data_dict = {
        1: "src/scripts/dataset1.sh",
        2: "src/scripts/dataset2.sh",
        3: "src/scripts/dataset3.sh",
        4: "src/scripts/dataset4.sh",
    }
    if not os.path.exists(os.path.join("data", "1_extracted", f"dataset{dataset_id}")):
        os.system(f"sh {data_dict[dataset_id]}")
