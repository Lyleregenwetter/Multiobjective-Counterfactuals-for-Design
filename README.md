# Multiobjective-Counterfactuals-for-Design

[//]: # (Official Repository for Multi-Objective Counterfactuals for Design &#40;MCD&#41;)

[//]: # ([![Contributors][contributors-shield]][contributors-url])

[//]: # ([![Forks][forks-shield]][forks-url])

[//]: # ([![Stargazers][stars-shield]][stars-url])

[//]: # ([![Issues][issues-shield]][issues-url])

[//]: # ([![MIT License][license-shield]][license-url])

[//]: # ([![LinkedIn][linkedin-shield]][linkedin-url])



<!-- PROJECT LOGO -->

[//]: # (<br />)
<div>

[//]: # (  <a href="https://github.com/Lyleregenwetter/Multiobjective-Counterfactuals-for-Design">)

[//]: # (    <img src="images/logo.png" alt="Logo" width="80" height="80">)

[//]: # (  </a>)

[//]: # (<h3 align="center">Multiobjective Counterfactuals for Design &#40;MCD&#41;</h3>)

  <p>
    MCD generates counterfactuals that meet multiple, customizable objectives in both the feature and performance spaces.  
    <br />

[//]: # (    <a href="https://github.com/Lyleregenwetter/Multiobjective-Counterfactuals-for-Design"><strong>Explore the docs »</strong></a>)

[//]: # (    <br />)

[//]: # (    <br />)

[//]: # (    <a href="https://github.com/Lyleregenwetter/Multiobjective-Counterfactuals-for-Design">View Demo</a>)

[//]: # (    ·)

[//]: # (    <a href="https://github.com/Lyleregenwetter/Multiobjective-Counterfactuals-for-Design/issues">Report Bug</a>)

[//]: # (    ·)

[//]: # (    <a href="https://github.com/Lyleregenwetter/Multiobjective-Counterfactuals-for-Design/issues">Request Feature</a>)
  </p>
</div>


[//]: # (<!-- TABLE OF CONTENTS -->)

[//]: # (<details>)

[//]: # (  <summary>Table of Contents</summary>)

[//]: # (  <ol>)

[//]: # (    <li>)

[//]: # (      <a href="#about-the-project">About The Project</a>)

[//]: # (      <ul>)

[//]: # (        <li><a href="#built-with">Built With</a></li>)

[//]: # (      </ul>)

[//]: # (    </li>)

[//]: # (    <li>)

[//]: # (      <a href="#getting-started">Quick-Start Guide</a>)

[//]: # (      <ul>)

[//]: # (        <li><a href="#prerequisites">Prerequisites</a></li>)

[//]: # (        <li><a href="#installation">Installation</a></li>)

[//]: # (      </ul>)

[//]: # (    </li>)

[//]: # (    <li><a href="#usage">Quick-Start Guide</a></li>)

[//]: # (    <li><a href="#roadmap">Roadmap</a></li>)

[//]: # (    <li><a href="#contributing">Contributing</a></li>)

[//]: # (    <li><a href="#license">License</a></li>)

[//]: # (    <li><a href="#contact">Contact</a></li>)

[//]: # (    <li><a href="#acknowledgments">Acknowledgments</a></li>)

[//]: # (  </ol>)

[//]: # (</details>)



<!-- ABOUT THE PROJECT -->

## About The Project

Multiobjective Counterfactuals for Design (MCD) is a framework primarily intended for the generation of design
alternatives that meet user-specified
performance criteria while remaining within a certain region of the design space. To use MCD, you need a dataset of
designs that are reasonably representative of the desired region of the design space, including performance metrics,
as well as a model capable of predicting the performance metrics of a given design. MCD is model agnostic - this means
that the model need not be a differentiable machine learning model, or,
in fact, a machine learning model in the first place. MCD also offers high flexibility in terms of the number and 'type'
of performance targets that can be specified.
Performance targets can be any combination of:

* 'Continuous Targets': (e.g. I want suggested bike designs to weigh between 2 and 4 kilograms)
* 'Classification Targets': (e.g. I want suggested bike designs to be classified as dirt bikes)
* 'Probability Targets': (e.g. I want each suggested design to have a higher probability of
  belonging to classes A or B than of
  belonging to C or D)

[//]: # ([![Product Name Screen Shot][product-screenshot]]&#40;https://example.com&#41;)

[//]: # ()

[//]: # (Here's a blank template to get started: To avoid retyping too much info. Do a search and replace with your text editor)

[//]: # (for the)

[//]: # (following: `github_username`, `repo_name`, `twitter_handle`, `linkedin_username`, `email_client`, `email`, `project_title`, `project_description`)

[//]: # (<p align="right">&#40;<a href="#readme-top">back to top</a>&#41;</p>)

### Built With

* [![Python][python-badge-url]][python-url]
* [![Pymoo][pymoo-badge-url]][pymoo-url]
* [![Pandas][pandas-badge-url]][pandas-url]
* [![Numpy][numpy-badge-url]][numpy-url]

[//]: # (<p align="right">&#40;<a href="#readme-top">back to top</a>&#41;</p>)



<!-- GETTING STARTED -->

## Quick-Start Guide

### Installation

1. Install MCD with:
   ```pip install decode-mcd```
2. Run:

```python
import random

from pymoo.core.variable import Real

import numpy as np
from decode_mcd import DesignTargets, DataPackage, MultiObjectiveProblem, CounterfactualsGenerator, ContinuousTarget

x = np.random.random(100)
x = x.reshape(100, 1)
y = x * 100 + random.random()


def predict(_x):
    return _x * 100


data_package = DataPackage(features_dataset=x,
                           predictions_dataset=y,
                           query_x=x[0].reshape(1, 1),
                           design_targets=DesignTargets([ContinuousTarget(label=0,
                                                                          lower_bound=25,
                                                                          upper_bound=75)]),
                           datatypes=[Real(bounds=(0, 1))])

problem = MultiObjectiveProblem(data_package=data_package,
                                prediction_function=lambda design: predict(design),
                                constraint_functions=[])

generator = CounterfactualsGenerator(problem=problem,
                                     pop_size=10,
                                     initialize_from_dataset=False)

generator.generate(n_generations=10)
counterfactuals = generator.sample_with_dtai(num_samples=10, gower_weight=1,
                                             avg_gower_weight=1, cfc_weight=1,
                                             diversity_weight=50)
print(counterfactuals)
```

[//]: # (<p align="right">&#40;<a href="#readme-top">back to top</a>&#41;</p>)



<!-- USAGE EXAMPLES -->

[//]: # (## I-Got-Time Guide)

[//]: # ()
[//]: # (1. Either install MCD with pip as shown in the Quick-Start Guide, or fork the repo with)

[//]: # (   ```git clone git@github.com:Lyleregenwetter/Multiobjective-Counterfactuals-for-Design.git```)

[//]: # (2. Now, customize the code below to fit your datasets and model. The template below assumes the following:)

[//]: # (    * The features_dataset _X_ has 4 columns: R1, R2, C1, in order. R1 and R2 are real variables)

[//]: # (      with the following respective ranges &#40;0, 10&#41; and &#40;-50, 50&#41;. C1 is a choice variable &#40;0, 1, 2&#41;.)

[//]: # (    * The predictions_dataset _Y_ has 5 columns. O_R1 and O_R2 are real variables.)

[//]: # (      O_C1 is a categorical/choice variable. O_P1 and O_P2 represent the probabilities of belonging to classes A and B,)

[//]: # (      respectively, where a design can belong to either class A or B and nothing else.)

[//]: # (    *)

[//]: # ()
[//]: # (```python)

[//]: # (from pymoo.core.variable import Real, Choice)

[//]: # (from decode_mcd import DesignTargets, DataPackage, MultiObjectiveProblem, CounterfactualsGenerator, ContinuousTarget)

[//]: # ()
[//]: # (x, y = ...  # load your data)

[//]: # (model = ...  # load your model)

[//]: # (query_x = ...  # define the initial design or starting point)

[//]: # ()
[//]: # (data_package = DataPackage&#40;features_dataset=x,)

[//]: # (                           predictions_dataset=y,)

[//]: # (                           query_x=query_x,)

[//]: # (                           design_targets=DesignTargets&#40;[ContinuousTarget&#40;label=0,)

[//]: # (                                                                          lower_bound=25,)

[//]: # (                                                                          upper_bound=75&#41;]&#41;,)

[//]: # (                           datatypes=[Real&#40;bounds=&#40;0, 10&#41;&#41;,)

[//]: # (                                      Real&#40;bounds=&#40;-50, 50&#41;&#41;,)

[//]: # (                                      Choice&#40;options=[0, 1, 2]&#41;],)

[//]: # (                           # # optional parameters)

[//]: # (                           # features_to_vary=..., )

[//]: # (                           # bonus_objectives=...,)

[//]: # (                           # datasets_validity=...,)

[//]: # (                           # datasets_scores=...,)

[//]: # (                           &#41;)

[//]: # ()
[//]: # (problem = MultiObjectiveProblem&#40;data_package=data_package,)

[//]: # (                                prediction_function=lambda design: model.predict&#40;design&#41;,)

[//]: # (                                constraint_functions=[]&#41;)

[//]: # ()
[//]: # (generator = CounterfactualsGenerator&#40;problem=problem,)

[//]: # (                                     pop_size=10,)

[//]: # (                                     initialize_from_dataset=False,)

[//]: # (                                     verbose=True&#41;)

[//]: # ()
[//]: # (generator.generate&#40;n_generations=10&#41;)

[//]: # (counterfactuals = generator.sample_with_dtai&#40;num_samples=10, gower_weight=1,)

[//]: # (                                             avg_gower_weight=1, cfc_weight=1,)

[//]: # (                                             diversity_weight=50&#41;)

[//]: # (print&#40;counterfactuals&#41;)

[//]: # (```)

<!-- ROADMAP -->

## Roadmap

- [ ] We are currently working on support gradient-based optimization



See the [open issues](https://github.com/Lyleregenwetter/Multiobjective-Counterfactuals-for-Design/issues) for a full
list of proposed features (and
known issues).

[//]: # (<p align="right">&#40;<a href="#readme-top">back to top</a>&#41;</p>)



<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any
contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also
simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

[//]: # (<p align="right">&#40;<a href="#readme-top">back to top</a>&#41;</p>)



<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE` for more information.

[//]: # (<p align="right">&#40;<a href="#readme-top">back to top</a>&#41;</p>)


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge

[contributors-url]: https://github.com/Lyleregenwetter/Multiobjective-Counterfactuals-for-Design/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge

[forks-url]: https://github.com/Lyleregenwetter/Multiobjective-Counterfactuals-for-Design/network/members

[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge

[stars-url]: https://github.com/Lyleregenwetter/Multiobjective-Counterfactuals-for-Design/stargazers

[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge

[issues-url]: https://github.com/Lyleregenwetter/Multiobjective-Counterfactuals-for-Design/issues

[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge

[license-url]: https://github.com/Lyleregenwetter/Multiobjective-Counterfactuals-for-Design/blob/master/LICENSE


[python-badge-url]: https://img.shields.io/badge/language-python-purple

[python-url]: https://www.python.org/

[pandas-badge-url]: https://img.shields.io/badge/framework-pandas-red

[pandas-url]: https://pandas.pydata.org/

[numpy-badge-url]: https://img.shields.io/badge/frameowrk-numpy-green

[numpy-url]: https://numpy.org/

[pymoo-badge-url]: https://img.shields.io/badge/framework-pymoo-blue

[pymoo-url]: https://pymoo.org/
