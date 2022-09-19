file_name = "model-best.h5"
if config["finetune"]:
    if os.path.exists(file_name):
        os.remove("model-best.h5")

    api = wandb.Api()
    run = api.run(
        config["pretrained_model_link"]
    )  # different-sweep-34-efficientnet-epoch-3-val_f1_score-0.71.hdf5
    run.file(file_name).download()
    model = tf.keras.models.load_model(file_name)
    pml = config["pretrained_model_link"]
    print(f"Downloaded Trained model: {pml},\nfinetuning...")
else:
    # build model
    model = get_model(config)
