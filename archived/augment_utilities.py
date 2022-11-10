import matplotlib.pyplot as plt
from tensorflow import cast, float32


def visualize_dataset(dataset, title):
    plt.figure(figsize=(12, 12)).suptitle(title, fontsize=18)
    for i, samples in enumerate(iter(dataset.take(9))):
        images = samples["images"]
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[0].numpy().astype("uint8"))
        plt.axis("off")
    plt.show()


# Notice that we use label_smoothing=0.1 in the loss function.
# When using MixUp, label smoothing is highly recommended.
def apply_rand_augment(inputs):
    import keras_cv

    rand_augment = keras_cv.layers.RandAugment(
        value_range=(0, 255),
        augmentations_per_image=3,
        magnitude=0.3,
        magnitude_stddev=0.2,
        rate=0.5,
    )
    inputs["images"] = rand_augment(inputs["images"])
    return inputs


def cut_mix_and_mix_up(samples):
    import keras_cv

    cut_mix = keras_cv.layers.CutMix()
    mix_up = keras_cv.layers.MixUp()
    samples = cut_mix(samples, training=True)
    samples = mix_up(samples, training=True)
    return samples


def preprocess_for_model(inputs):
    images, labels = inputs["images"], inputs["labels"]
    images = cast(images, float32) / 255.0
    return images, labels


#| export

def download_configs():
    config_files = {
        1:{'url':'https://raw.githubusercontent.com/udaylunawat/Whats-this-rock/main/configs/bad_images%20selected%20by%20gemini.txt',
           'file_name':'bad_images selected by gemini.txt'},
        2:{'url':'https://raw.githubusercontent.com/udaylunawat/Whats-this-rock/main/configs/config.yaml',
           'file_name':'config.yaml'},
        3:{'url':'https://raw.githubusercontent.com/udaylunawat/Whats-this-rock/main/configs/duplicates%20selected%20by%20gemini.txt',
           'file_name':'duplicates selected by gemini.txt'},
        4:{'url':'https://raw.githubusercontent.com/udaylunawat/Whats-this-rock/main/configs/misclassified%20selected%20by%20gemini.txt',
           'file_name':'misclassified selected by gemini.txt'},
        5:{'url':'https://raw.githubusercontent.com/udaylunawat/Whats-this-rock/main/configs/sweep.yaml','file_name':'sweep.yaml'}
    }
    def download_file(url, file_name, dest_dir='configs'):
        """Download and write file to destination directory."""
        r = requests.get(url, allow_redirects=True)
        open(os.path.join(dest_dir, file_name), 'wb').write(r.content)
    
    os.makedirs('configs', exist_ok=True)
    for index, file in config_files.items():
        file_url = file['url']
        file_name = file['file_name']
        download_file(file_url, file_name)
        
        