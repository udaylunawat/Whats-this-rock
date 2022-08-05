# Whats-this-rock
## Rock Classification Telegram Bot!
![code-size](https://img.shields.io/github/languages/code-size/udaylunawat/Whats-this-rock) ![repo-size](https://img.shields.io/github/repo-size/udaylunawat/Whats-this-rock) ![top-language](https://img.shields.io/github/languages/top/udaylunawat/Whats-this-rock)
<p align="left">
    <img src="imgs/marie.jpg " alt="What's my name?" width="200"/>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

- [Table of Contents](#table-of-contents)
<!-- - [Motivation](#motivation) -->
- [Getting Started](#getting-started)
  - [Installation](#installation)
- [Notebooks](#instructions)

- [Usage](#usage)
- [Technologies Used](#technologies-used)
<!-- - [Roadmap](#roadmap) -->
- [Contributing](#contributing)
- [License](#license)
<!-- - [Contact](#contact) -->
# Instructions
- Paste your kaggle.json file here or make sure it's in the root directory
- Run the commands below
```
pip install -r requirements.txt
sh setup.sh
python preprocess.py --root data/1_extracted/Rock_Dataset/ \
                      --remove_class minerals \
                      --oversample
python train.py
```

## Notebooks
<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/drive/1N1CIqdOKlJSJla5PU53Yn9KWSao47eMv?usp=sharing"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/udaylunawat/Whats-this-rock"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
  <td>
    <a href="https://github.com/udaylunawat/Whats-this-rock/blob/main/notebook.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
</table>

<br></br>

## Bug / Feature Request
If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an issue [here](https://github.com/prasanna) by including your search query and the expected result.

If you'd like to request a new function, feel free to do so by opening an issue [here](https://github.com/). Please include sample queries and their corresponding results.


## Technologies Used
- [Google Colab](https://colab.research.google.com/?utm_source=scs-index)
- [Python](https://www.python.org/)
- [Tensorflow](https://www.tensorflow.org/)
- [Weights and Biases](https://wandb.ai/site)
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## License
[![Apache license](https://img.shields.io/badge/license-apache-blue?style=for-the-badge&logo=appveyor)](http://www.apache.org/licenses/LICENSE-2.0e)

Copyright 2020 Uday

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Credits
- [Dataset - by Mahmoud Alforawi](https://www.kaggle.com/datasets/mahmoudalforawi/igneous-metamorphic-sedimentary-rocks-and-minerals)

<br>
<p align="center"> Walt might be the one who knocks <br> but Hank is the one who rocks. </br> </p>

[contributors-shield]: https://img.shields.io/github/contributors/udaylunawat/Covid-19-Radiology.svg?style=flat-square
[contributors-url]: https://github.com/udaylunawat/Covid-19-Radiology/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/udaylunawat/Covid-19-Radiology.svg?style=flat-square
[forks-url]: https://github.com/udaylunawat/Covid-19-Radiology/network/members

[stars-shield]: https://img.shields.io/github/stars/udaylunawat/Covid-19-Radiology.svg?style=flat-square
[stars-url]: https://github.com/udaylunawat/Covid-19-Radiology/stargazers

[issues-shield]: https://img.shields.io/github/issues/udaylunawat/Covid-19-Radiology.svg?style=flat-square
[issues-url]: https://github.com/udaylunawat/Covid-19-Radiology/issues

[license-shield]: https://img.shields.io/github/license/udaylunawat/Covid-19-Radiology.svg?style=flat-square
[license-url]: https://github.com/udaylunawat/Covid-19-Radiology/blob/master/LICENSE

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/uday-lunawat
