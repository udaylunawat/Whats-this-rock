import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="rocks",
    version="0.0.1",
    author="Uday Lunawat",
    author_email="udaylunawat@gmail.com",
    description=(
        "Rock classifier deployed on heroku and monitored using Weights and Biases."
    ),
    license="MIT License",
    keywords="image_classification tensorflow keras wandb telegram-bot",
    packages=["src", "configs"],
    long_description=read("README.md"),
)
