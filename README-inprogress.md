# Multiobjective-Counterfactuals-for-Design

Official Repository for Multi-Objective Counterfactuals for Design (MCD)

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">

[//]: # (  <a href="https://github.com/Lyleregenwetter/Multiobjective-Counterfactuals-for-Design">)

[//]: # (    <img src="images/logo.png" alt="Logo" width="80" height="80">)

[//]: # (  </a>)

<h3 align="center">Multiobjective Counterfactuals for Design (MCD)</h3>

  <p align="center">
    MCD generates counterfactuals that meet multiple, customizable objectives in both the feature and performance spaces.  
    <br />
    <a href="https://github.com/Lyleregenwetter/Multiobjective-Counterfactuals-for-Design"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Lyleregenwetter/Multiobjective-Counterfactuals-for-Design">View Demo</a>
    ·
    <a href="https://github.com/Lyleregenwetter/Multiobjective-Counterfactuals-for-Design/issues">Report Bug</a>
    ·
    <a href="https://github.com/Lyleregenwetter/Multiobjective-Counterfactuals-for-Design/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Quick-Start Guide</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

## About The Project

Multiobjective Counterfactuals for Design (MCD) is a framework primarily intended for the generation of design
alternatives that meet user-specified
performance criteria while remaining within a certain region of the design space. To use MCD, you need a dataset of
designs that are reasonably representative of the desired region of the design space, including performance metrics,
as well as a model capable of predicting the performance metrics of a given design. MCD is model agnostic - this means
that the model need not be gradient based or,
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

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Python][python-badge-url]][python-url]
* [![Pymoo][pymoo-badge-url]][pymoo-url]
* [![Pandas][pandas-badge-url]][pandas-url]
* [![Numpy][numpy-badge-url]][numpy-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->

## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Installation

You can install MCD with:
```pip install decode-mcd```

Alternatively,

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->

## Quick-Start Guide

```python
import random
from pymoo.core.variable import Real
from data_package import DataPackage
from design_targets import *
from multi_objective_cfe_generator import MultiObjectiveCounterfactualsGenerator, CFSet

x = np.random.random(100)
x = x.reshape(100, 1)
y = x * 100 + random.random()


def predict(_x):
    return _x * 100


data_package = DataPackage(x, y, x[0].reshape(1, 1),
                           design_targets=DesignTargets([ContinuousTarget(label=0,
                                                                          lower_bound=25,
                                                                          upper_bound=75)]),
                           datatypes=[Real(bounds=(0, 1))])
gen = MultiObjectiveCounterfactualsGenerator(data_package, lambda design: predict(design), [])
cf_set = CFSet(gen, 10, False)
cf_set.optimize(10)
counterfactuals = cf_set.sample(10, 1, 1, 1, 1, 50)
print(counterfactuals)
```

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos
work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->

## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/Lyleregenwetter/Multiobjective-Counterfactuals-for-Design/issues) for a full
list of proposed features (and
known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



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

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->

## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project
Link: [https://github.com/Lyleregenwetter/Multiobjective-Counterfactuals-for-Design](https://github.com/Lyleregenwetter/Multiobjective-Counterfactuals-for-Design)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>



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

[license-url]: https://github.com/Lyleregenwetter/Multiobjective-Counterfactuals-for-Design/blob/master/LICENSE.txt

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555

[linkedin-url]: https://linkedin.com/in/linkedin_username

[product-screenshot]: images/screenshot.png

[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white

[Next-url]: https://nextjs.org/

[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB

[React-url]: https://reactjs.org/

[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D

[Vue-url]: https://vuejs.org/

[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white

[Angular-url]: https://angular.io/

[python-badge-url]: https://img.shields.io/badge/language-python-purple

[python-url]: https://www.python.org/

[pandas-badge-url]: https://img.shields.io/badge/framework-pandas-red

[pandas-url]: https://pandas.pydata.org/

[numpy-badge-url]: https://img.shields.io/badge/frameowrk-numpy-green

[numpy-url]: https://numpy.org/

[pymoo-badge-url]: https://img.shields.io/badge/framework-pymoo-blue

[pymoo-url]: https://pymoo.org/

[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00

[Svelte-url]: https://svelte.dev/

[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white

[Laravel-url]: https://laravel.com

[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white

[Bootstrap-url]: https://getbootstrap.com

[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white

[JQuery-url]: https://jquery.com 