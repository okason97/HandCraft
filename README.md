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
        <li><a href="#datasets">Datasets</a></li>
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

Generate and classify sign language gestures with HandCraft. The steps below go from a fresh clone to trained and tested models. Each step links to the full documentation in [doc/](doc/README.md).

### Installation

1. Clone the repo and install the dependencies with [uv](https://docs.astral.sh/uv/getting-started/installation/)
   ```sh
   git clone https://github.com/okason97/HandCraft.git
   cd HandCraft
   uv sync --all-extras
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```
2. Choose where the datasets and the outputs go
   ```sh
   export HANDCRAFT_DATA=/path/to/datasets
   export HANDCRAFT_SAVE=/path/to/outputs
   export WANDB_MODE=offline   # or run `wandb login` to upload the training curves
   ```
   The two directories can also be passed to the run scripts with `--data-root <dir>` and `--save-root <dir>`.

More in [doc/installation.md](doc/installation.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Datasets

Models are trained on keypoints. LSFB provides them; for INCLUDE and DiSPLaY they are extracted from the videos with MediaPipe.

  LSFB
   ```sh
   python scripts/data/LSFB/setup_lsfb.py -data_dir $HANDCRAFT_DATA
   ```
  INCLUDE (57 GB of videos; the keypoint extraction takes several hours)
   ```sh
   ./scripts/data/download_mediapipe_models.sh $HANDCRAFT_DATA/mediapipe
   ./scripts/data/INCLUDE/download_data.sh $HANDCRAFT_DATA/INCLUDE
   python scripts/data/INCLUDE/format.py -data_dir $HANDCRAFT_DATA/INCLUDE -model_dir $HANDCRAFT_DATA/mediapipe
   python scripts/data/INCLUDE/make_official_split.py -data_dir $HANDCRAFT_DATA/INCLUDE -out_dir $HANDCRAFT_DATA/INCLUDE_official
   ```
  DiSPLaY: download the `Signs(...).zip` files from [IEEE DataPort](https://doi.org/10.21227/5gsb-fb69) (needs an account), extract them into `$HANDCRAFT_DATA/DiSPLaY/original/`, then
   ```sh
   python scripts/data/DiSPLaY/format.py -data_dir $HANDCRAFT_DATA/DiSPLaY -model_dir $HANDCRAFT_DATA/mediapipe
   ```

More in [doc/datasets.md](doc/datasets.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Every run is `<script> <mode> <model> <config> <dataset>`, with the config in `src/configs/<dataset>/<model>/<config>.yaml`. Add `--seed` to make a run reproducible.

### Sign Language Recognition

  Train, validating after every epoch
   ```sh
   ./scripts/run/train.sh classification ViT original-pad-128x2 INCLUDE --seed 42
   ```
  Train and evaluate the best checkpoint on the test set
   ```sh
   ./scripts/run/test.sh classification ViT original-pad-128x2 INCLUDE --seed 42
   ```
  Evaluate a trained model on the test set
   ```sh
   ./scripts/run/eval.sh classification ViT original-pad-128x2 INCLUDE -ckpt $HANDCRAFT_SAVE/INCLUDE/ViT-original-pad-128x2/checkpoints/<run name>/
   ```
  Our best model on the official INCLUDE split (94.3% top-1, see [doc/results-include.md](doc/results-include.md))
   ```sh
   ./scripts/run/test.sh classification stgcn official-lr5 INCLUDE -data $HANDCRAFT_DATA/INCLUDE_official/ --seed 42
   ```

### Sign Language Generation

  Train the generator, validating after every epoch (use `scripts/run/test.sh` to also evaluate on the test set)
   ```sh
   ./scripts/run/train.sh cond_prediction CsiMLPe depth_big_noise_0.1 INCLUDE --seed 42
   ./scripts/run/train.sh cond_prediction CsiMLPe depth_big_noise_0.1-reversed INCLUDE --reverse --seed 42
   ```
  Generate a synthetic dataset with both generators
   ```sh
   ./scripts/run/generate_dataset.sh cond_prediction CsiMLPe depth_big_noise_0.1-reversed INCLUDE --sd_num 100 --seed 42 \
       -ckpt $HANDCRAFT_SAVE/INCLUDE/CsiMLPe-depth_big_noise_0.1/checkpoints/<forward run name>/ \
       -tg -r_ckpt $HANDCRAFT_SAVE/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/checkpoints/<reversed run name>/
   ```
  Pretrain a recognition model on it, then train and test on the real data
   ```sh
   ./scripts/run/test.sh classification ViT original-pad-synth75-475 INCLUDE --seed 42 \
       -s_data $HANDCRAFT_SAVE/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/<generated dataset>
   ```
  Example of a generated sequence:
![Example][example]

Logs are written to `logs/<dataset>/<model>-<config>/`. During training the logged metrics are **validation** metrics; the test result is the `Test` line after `End of training!`. More in [doc/training.md](doc/training.md) and [doc/synthetic-data.md](doc/synthetic-data.md).

### Documentation

| | |
|---|---|
| [Installation](doc/installation.md) | [Datasets](doc/datasets.md) |
| [Training, validation and testing](doc/training.md) | [Synthetic data pretraining](doc/synthetic-data.md) |
| [Configuration reference](doc/configuration.md) | [Models](doc/models.md) |
| [Results on the official INCLUDE split](doc/results-include.md) | [Reproducibility and known issues](doc/reproducibility.md) |
| [Development](doc/development.md) | |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Publication -->
## Publication

**Paper**: [HandCraft: Dynamic Sign Generation for Synthetic Data Augmentation](https://arxiv.org/abs/2508.14345)

### Citation

If you use this work in your research, please cite:
```bibtex
@misc{rios2025handcraftdynamicsigngeneration,
      title={HandCraft: Dynamic Sign Generation for Synthetic Data Augmentation}, 
      author={Gaston Gustavo Rios and Pedro Dal Bianco and Franco Ronchetti and Facundo Quiroga and Oscar Stanchi and Santiago Ponte Ahón and Waldo Hasperué},
      year={2025},
      eprint={2508.14345},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.14345}, 
}
```

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Gaston Rios - okason1997@hotmail.com

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
