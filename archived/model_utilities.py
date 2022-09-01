def get_model(config):
    models_dict = {
        "efficientnet": models.get_efficientnet,
        "resnet": models.get_resnet,
        "small_cnn": models.get_small_cnn,
        "large_cnn": models.get_large_cnn,
        "baseline": models.get_baseline,
        "mobilenet": models.get_mobilenet,
        "mobilenetv2": models.get_mobilenetv2,
        "inceptionresnetv2": models.get_inceptionresnetv2,
        "efficientnetv2m": models.get_efficientnetv2m,
    }

    return models_dict[config.model_config.backbone](config)


def get_best_checkpoint():
    max_val_acc = 0
    best_model = None
    for file_name in os.listdir("checkpoints"):
        if file_name.endswith("5"):
            val_acc = int(os.path.basename(file_name).split(".")[-2])
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                best_model = file_name
    return best_model


# https://datamonje.com/image-data-augmentation/#cutout
# https://colab.research.google.com/drive/1on9rvQdr0s8CfqYeZBTbgk0K5I_Dha-X#scrollTo=py9wqsM05kio
def custom_augmentation(np_tensor):
    def random_contrast(np_tensor):
        return np.array(image.random_contrast(np_tensor, 0.5, 2))

    def random_hue(np_tensor):
        return np.array(image.random_hue(np_tensor, 0.5))

    def random_saturation(np_tensor):
        return np.array(image.random_saturation(np_tensor, 0.2, 3))

    def random_crop(np_tensor):
        # cropped height between 70% to 130% of an original height
        new_height = int(np.random.uniform(0.7, 1.30) * np_tensor.shape[0])
        # cropped width between 70% to 130% of an original width
        new_width = int(np.random.uniform(0.7, 1.30) * np_tensor.shape[1])
        # resize to new height and width
        cropped = image.resize_with_crop_or_pad(np_tensor, new_height,
                                                new_width)
        return np.array(image.resize(cropped, np_tensor.shape[:2]))

    def gaussian_noise(np_tensor):
        mean = 0
        # variance: randomly between 1 to 25
        var = np.random.randint(1, 26)
        # sigma is square root of the variance value
        noise = np.random.normal(mean, var**0.5, np_tensor.shape)
        return np.clip(np_tensor + noise, 0, 255).astype("int")

    def cutout(np_tensor):
        cutout_height = int(np.random.uniform(0.1, 0.2) * np_tensor.shape[0])
        cutout_width = int(np.random.uniform(0.1, 0.2) * np_tensor.shape[1])
        cutout_height_point = np.random.randint(np_tensor.shape[0] -
                                                cutout_height)
        cutout_width_point = np.random.randint(np_tensor.shape[1] -
                                               cutout_width)
        np_tensor[cutout_height_point:cutout_height_point + cutout_height,
                  cutout_width_point:cutout_width_point +
                  cutout_width, :, ] = 127
        return np_tensor

    if np.random.uniform() < 0.1:
        np_tensor = random_contrast(np_tensor)
    if np.random.uniform() < 0.1:
        np_tensor = random_hue(np_tensor)
    if np.random.uniform() < 0.1:
        np_tensor = random_saturation(np_tensor)
    if np.random.uniform() < 0.2:
        np_tensor = random_crop(np_tensor)

    # Gaussian noise giving error hence removed
    # if (np.random.uniform() < 0.2):
    #     np_tensor = gaussian_noise(np_tensor)
    if np.random.uniform() < 0.3:
        np_tensor = cutout(np_tensor)
    return np.array(np_tensor)