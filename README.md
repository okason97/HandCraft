<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

![Product Name Screen Shot][product-screenshot]

Sign language generation and recognition using synthetic data augmentation. Repository for the models and experiments used in the paper **Sign Generation for Data Augmentation**.

Conditional Human motion prediction model:
* CMLPe

Sign language recognition models:
* Mamba
* Transformer

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

Generate and classify sign language gestures with HandCraft. To do it follow these steps:

### Installation

Clone the repo
   ```sh
   git clone https://github.com/okason97/HandCraft.git
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

1. Create a config file. Examples in /src/configs
2. Use /script.sh to use the models. 
   ```sh
  # Sign Language Recognition
   ./script.sh classification mamba original128-pad LSFB

  # Sign Language Generation
   ./script.sh cond_prediction CsiMLPe depth_big_noise_0.1
   ```
  Example of generated sequence:
![Example][example]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Gaston Rios - okason1997@hotmail.com

Project Link: [https://github.com/okason97/HandCraft](https://github.com/okason97/HandCraft)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/okason97/HandCraft.svg?style=for-the-badge
[contributors-url]: https://github.com/okason97/HandCraft/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/okason97/HandCraft.svg?style=for-the-badge
[forks-url]: https://github.com/okason97/HandCraft/network/members
[stars-shield]: https://img.shields.io/github/stars/okason97/HandCraft.svg?style=for-the-badge
[stars-url]: https://github.com/okason97/HandCraft/stargazers
[issues-shield]: https://img.shields.io/github/issues/okason97/HandCraft.svg?style=for-the-badge
[issues-url]: https://github.com/okason97/HandCraft/issues
[license-shield]: https://img.shields.io/github/license/okason97/HandCraft.svg?style=for-the-badge
[license-url]: https://github.com/okason97/HandCraft/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/gaston-gustavo-rios/
[product-screenshot]: images/graphical_abstract.png
[example]: images/value_keypoints_0.gif
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 