# Interaction Feature-Guided Explainable Particle Chebyshev Networks (E-PCN) for Jet Tagging

<p align="center">
  <a href="https://arxiv.org/abs/2512.07420">
    <img src="https://img.shields.io/badge/arXiv-2512.07420-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"/>
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/Submitted%20to-JHEP-006699?style=for-the-badge&logo=academia&logoColor=white" alt="JHEP"/>
</p>


<p align="center">
  <a href="https://ccds.ai/">
    <img src="https://img.shields.io/badge/Maintained%20by-CCDS%20Team-ff8c00?style=for-the-badge&logo=academia&logoColor=white" alt="Maintained by CCDS"/>
  </a>
  &nbsp;
  <a href="https://github.com/Adrita-Khan/Jet-Tagging/tree/main">
    <img src="https://img.shields.io/badge/Exploratory%20Repo-Full%20Experiments%20%26%20Ablations-2ea44f?style=for-the-badge&logo=github&logoColor=white" alt="Full Repository"/>
  </a>
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
  <a href="https://github.com/Adrita-Khan/Jet-Tagging/issues"><img src="https://img.shields.io/github/issues/Adrita-Khan/Jet-Tagging" alt="Issues"></a>
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=800&color=ff8c00&center=true&vCenter=true&repeat=true&width=820&lines=Jet+Tagging+%7C+High-Energy+Physics;Particle+Chebyshev+Networks+(PCN);Physics-Motivated+Feature+Engineering;Lund+Jet+Plane+Inspired;4-Momentum+Interaction+Features;JetClass+Dataset+%7C+CERN+Open+Data;QCD-Informed+Dependencies;Deep+Learning+for+Particle+Physics;Jet+Substructure+Analysis;Graph+Neural+Networks;Explainable+AI+in+HEP" alt="Typing SVG" />
</p>

> **Note:** This project is ongoing and subject to continuous updates.

---

## Overview

This repository presents **E-PCN** — an enhanced **Particle Chebyshev Network** for jet tagging in high-energy physics. Jet tagging refers to the task of classifying collimated sprays of particles (jets) produced in high-energy collisions and associating them with their originating particles or decay processes.

We augment the base PCN architecture with **physics-motivated interaction features** derived from particle 4-momentum vectors and inspired by the **Lund jet plane** formalism. These features bias the model toward fine-grained, QCD-informed inter-particle dependencies, improving discrimination capability for jet classification on the **JetClass dataset**.

This is an ongoing project of the [Center for Computational and Data Sciences (CCDS)](https://ccds.ai/) in collaboration with the [Department of Theoretical Physics, University of Dhaka](https://www.du.ac.bd/body/MissionVision/TPHY), with ties to [CERN](https://home.cern/).

---

## Highlights

- **Physics-informed features** derived from particle 4-momenta (angular separation, transverse momentum scale, momentum fraction, invariant mass)
- **Lund jet plane** inspired feature construction following Dreyer & Qu (2021)
- **End-to-end training** on CERN Open Data (JetClass dataset)
- **Evaluation** using standard HEP classification metrics

---

## Physics-Motivated Interaction Features

For each pair of particles $(a, b)$ in a jet, we compute four kinematic observables that capture key aspects of jet substructure. Because these variables typically exhibit long-tail distributions in high-energy physics, we apply a logarithmic transformation and use $(\ln \Delta,\ \ln k_T,\ \ln z,\ \ln m^2)$ as the interaction features fed to the network.

### Feature Definitions

| Feature | Formula | Description |
|:-------:|:-------:|:------------|
| $\Delta$ | $\Delta = \sqrt{(y_a - y_b)^2 + (\phi_a - \phi_b)^2}$ | Angular separation in the rapidity–azimuth plane |
| $k_T$ | $k_T = \min(p_{T,a},\, p_{T,b}) \cdot \Delta$ | Transverse momentum scale; captures soft/collinear structure |
| $z$ | $z = \dfrac{\min(p_{T,a},\, p_{T,b})}{p_{T,a} + p_{T,b}}$ | Momentum fraction; measures energy sharing between the pair |
| $m^2$ | $m^2 = (E_a + E_b)^2 - \lVert \mathbf{p}_a + \mathbf{p}_b \rVert^2$ | Squared invariant mass of the particle pair |

### Symbol Reference

| Symbol | Definition |
|:------:|:-----------|
| $y_i$ | Rapidity of particle $i$ |
| $\phi_i$ | Azimuthal angle of particle $i$ |
| $p_{T,i}$ | Transverse momentum: $p_{T,i} = \sqrt{p_{x,i}^2 + p_{y,i}^2}$ |
| $\mathbf{p}_i$ | Momentum 3-vector of particle $i$: $p_{i} = (p_{x,i}, p_{y,i}, p_{z,i})$ |
| $E_i$ | Energy of particle $i$ |
| $\lVert \cdot \rVert$ | Euclidean norm |

### Physical Motivation

This choice of features follows the Lund jet plane framework introduced by Dreyer & Qu (2021), which provides a QCD-grounded representation of jet splittings. By encoding angular separation, transverse momentum scale, energy sharing, and invariant mass, the model is guided toward the physically relevant structure of parton showers and hadronization — improving both discriminative performance and interpretability.

> Dreyer & Qu (2021). *Jet tagging in the Lund plane with graph networks.* [arXiv:2012.08526](https://arxiv.org/abs/2012.08526)

---

## Getting Started

### Installation

```bash
git clone https://github.com/Adrita-Khan/Jet-Tagging.git
cd Jet-Tagging
pip install -r requirements.txt
```

### Repository Structure

```
Jet-Tagging/
├── raqib-pcn-experiments/                              # Main experiment scripts and notebooks
├── pythia-data-gen.md                                  # Data generation tutorial for Pythia
├── pythia-installation.md                              # Pythia installation guide
├── pythia-jet-tagging-data-generation-tutorial.md      # Jet tagging data generation walkthrough
├── pythia-python-guide.md                              # Python interface guide for Pythia
├── requirements.txt                                    # Python dependencies
├── .gitattributes                                      # Git configuration
└── README.md                                           # This file
```

---

## References

| # | Title | Venue |
|:-:|:------|:-----:|
| 1 | JetClass: A Large-Scale Dataset for Deep Learning in Jet Physics | [JHEP 2024](https://link.springer.com/article/10.1007/JHEP07(2024)247) |
| 2 | Particle Chebyshev Network (PCN) | [PMLR (ICML 2022)](https://proceedings.mlr.press/v162/qu22b.html) |
| 3 | The Lund Jet Plane | [JHEP 2018](https://link.springer.com/article/10.1007/JHEP12(2018)064) |
| 4 | Jet Substructure and Machine Learning | [CPC 2024](https://iopscience.iop.org/article/10.1088/1674-1137/ad7f3d/meta) |
| 5 | Jet Tagging via Particle Clouds | [Phys. Rev. D 2020](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.056019) |
| 6 | Jet Tagging in the Lund Plane with Graph Networks | [arXiv:2012.08526](https://arxiv.org/abs/2012.08526) |
| 7 | PCN-Jet-Tagging (baseline implementation) | [GitHub](https://github.com/YVSemlani/PCN-Jet-Tagging) |

---

## Acknowledgements

We thank the [CERN Open Data Portal](http://opendata.cern.ch/) for providing high-quality collision data, and the original PCN authors for the base architecture. This work is supported by the [Center for Computational and Data Sciences (CCDS)](https://ccds.ai/).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

<p align="center">
  <strong>Questions or collaborations?</strong><br><br>
  <a href="mailto:adrita.khan.official@gmail.com"><img src="https://img.shields.io/badge/Adrita%20Khan-Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email Adrita Khan"/></a>
  &nbsp;
  <a href="https://www.linkedin.com/in/adrita-khan"><img src="https://img.shields.io/badge/Adrita%20Khan-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Adrita Khan"/></a>
  &nbsp;
  <a href="https://x.com/Adrita_"><img src="https://img.shields.io/badge/@Adrita__-Twitter%2FX-000000?style=for-the-badge&logo=x&logoColor=white" alt="Twitter Adrita"/></a>
  <br><br>
  <a href="mailto:raqibul.islam.academic@gmail.com"><img src="https://img.shields.io/badge/Md%20Raqibul%20Islam-Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email Raqibul Islam"/></a>
  &nbsp;
  <a href="https://www.linkedin.com/in/raqib03/"><img src="https://img.shields.io/badge/Md%20Raqibul%20Islam-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Raqibul Islam"/></a>
  &nbsp;
  <a href="https://x.com/raqib_03"><img src="https://img.shields.io/badge/@raqib__03-Twitter%2FX-000000?style=for-the-badge&logo=x&logoColor=white" alt="Twitter Raqibul"/></a>
</p>
