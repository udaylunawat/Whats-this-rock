# Whats-this-rock
Telegram Rock Classification Bot!

![God Damn it, Marie!](imgs/marie.jpg "Title")

# Instructions
- Paste your kaggle.json file in root directory
- Run the commands below in terminal
```
pip install -r requirements.txt
python preprocess.py
python efficientnet_train.py --project_name rock_classification \
                              --epochs 100 \
                              --notes "Efficient Net with f1 weighted" \
                              --learning_rate 0.0005 \
                              --sample_size 0.2 \
                              --batch_size 1024 \
                              --size 224
```
# To-do
- Improve preprocessing Techniques
- Dockerize
- Deploy on AWS
- Add notebook
- EDA
- Define proper metrics
- What makes a Good telegram bot?
- Best practices heroku or any deployment
- hyperparameter tuning
- Would procfiles work without port number?
    - As it is a worker app and not a web app
