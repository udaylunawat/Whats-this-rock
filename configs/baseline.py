import ml_collections
from ml_collections import config_dict


def get_wandb_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.project = "Whats-this-rock"
    configs.log_data_type = "train"
    return configs


def get_dataset_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.dataset_name = "rocks"
    configs.train_dataset = [1]
    configs.test_dataset = [0]
    configs.root = "data/1_extracted/"
    configs.sampling = None
    configs.image_height = 224
    configs.image_width = 224
    configs.channels = 3
    configs.batch_size = 64
    configs.num_classes = config_dict.placeholder(int)

    return configs


def get_model_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.model_img_channels = 3
    configs.backbone = "efficientnetv2m"
    configs.use_pretrained_weights = True
    configs.trainable = True
    configs.preprocess = True
    configs.dropout_rate = 0.4
    configs.post_gap_dropout = False

    return configs


def get_callback_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    # Early stopping
    configs.use_earlystopping = True
    configs.early_patience = 10
    # Reduce LR on plateau
    configs.use_reduce_lr_on_plateau = True
    configs.rlrp_factor = 0.4
    configs.rlrp_patience = 2
    configs.threshold = None
    configs.min_lr = 0.00001
    # Model checkpointing
    configs.checkpoint_filepath = "wandb/model_{epoch}"
    configs.save_model = False
    configs.save_best_only = True
    configs.monitor = 'val_f1_score'

    return configs


def get_train_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.epochs = 3
    configs.lr = 0.001
    configs.use_augmentations = True
    configs.use_class_weights = True
    configs.optimizer = "adamax"
    configs.loss = "categorical_crossentropy"
    configs.metrics = ["accuracy"]

    return configs


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.wandb_config = get_wandb_configs()
    config.dataset_config = get_dataset_configs()
    config.model_config = get_model_configs()
    config.callback_config = get_callback_configs()
    config.train_config = get_train_configs()

    return config
