# E-PCN: Jet Tagging with Explainable Particle Chebyshev Networks Using Kinematic Features

<p align="center">
  <a href="https://arxiv.org/abs/2512.07420">
    <img src="https://img.shields.io/badge/arXiv-2512.07420-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"/>
  </a>
  &nbsp;
  <a href="https://link.springer.com/journal/13130">
    <img src="https://img.shields.io/badge/Submitted%20to-JHEP-006699?style=for-the-badge&logo=academia&logoColor=white" alt="JHEP"/>
  </a>
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
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=800&color=ff8c00&center=true&vCenter=true&repeat=true&width=820&lines=Jet+Tagging+%7C+High-Energy+Physics;Explainable+Particle+Chebyshev+Networks;Physics-Motivated+Feature+Engineering;Lund+Jet+Plane+Inspired;4-Momentum+Interaction+Features;JetClass+Dataset+%7C+CERN+Open+Data;Grad-CAM+Explainability+in+HEP;Deep+Learning+for+Particle+Physics;Jet+Substructure+Analysis;Graph+Neural+Networks;Aspen+Open+Jets+%7C+Real+CMS+Data" alt="Typing SVG" />
</p>

> **Note:** This project is ongoing and subject to continuous updates.

---

## Overview

This repository presents **E-PCN** — the **Explainable Particle Chebyshev Network** — a graph neural network for jet tagging in high-energy physics. Jet tagging refers to the task of identifying and classifying collimated sprays of particles (jets) produced in high-energy collisions and associating them with their originating particles or decay processes.

E-PCN extends the base PCN architecture by constructing **four parallel graph representations** per jet, each weighted by a distinct physics-motivated kinematic variable: angular separation ($\Delta$), relative transverse momentum ($k_T$), momentum fraction ($z$), and invariant mass squared ($m^2$). Three of these variables ($\Delta$, $k_T$, $z$) are motivated by the **Lund jet plane** formalism, grounded in perturbative QCD factorization; the fourth ($m^2$) provides complementary mass-scale sensitivity for heavy-flavor jet identification. Explainability is achieved through **Gradient-weighted Class Activation Mapping (Grad-CAM)**, which quantifies each variable's contribution to classification decisions.

This is a project of the [Center for Computational and Data Sciences (CCDS)](https://ccds.ai/), Independent University, Bangladesh, in collaboration with the [Department of Theoretical Physics, University of Dhaka](https://www.du.ac.bd/body/MissionVision/TPHY).

---

## Key Results

### JetClass Benchmark (Simulated Data)

| Metric | PCN (baseline) | E-PCN (ours) | Improvement |
|:------:|:--------------:|:------------:|:-----------:|
| Macro-Accuracy | 0.9249 | **0.9467** | +2.36% |
| Macro-AUC | 0.9294 | **0.9678** | +4.13% |
| Macro-AUPR | 0.6599 | **0.8241** | +24.88% |

E-PCN achieves the highest classification accuracy among all compared models, surpassing the Particle Transformer (ParT) at 93.12% by 1.55 percentage points. The most dramatic gains are in heavy-flavor channels: AUPR for $H \to b\bar{b}$ improves by **81.53%** (0.4738 → 0.8601) and $H \to c\bar{c}$ by **51.54%** (0.4577 → 0.6936).

### State-of-the-Art Comparison

| Model | Macro-Accuracy | Macro-AUC |
|:------|:--------------:|:---------:|
| PFN | 0.8521 | 0.9103 |
| P-CNN | 0.8847 | 0.9312 |
| ParticleNet | 0.9015 | 0.9521 |
| ParT | 0.9312 | 0.9687 |
| PCN (baseline) | 0.9249 | 0.9294 |
| **E-PCN (ours)** | **0.9467** | **0.9678** |

### Real CMS Data: Aspen Open Jets

Evaluated on the Aspen Open Jets dataset of real CMS proton-proton collision data using unsupervised clustering metrics:

| Metric | PCN | E-PCN | Improvement |
|:------:|:---:|:-----:|:-----------:|
| Davies-Bouldin Index ↓ | 0.8395 | **0.4017** | −52.15% |
| Dunn Index ↑ | 0.0189 | **0.0269** | +42.33% |

---

## Highlights

- **Physics-informed multi-graph architecture**: Four parallel GNN branches, each processing a graph weighted by one of $\Delta$, $k_T$, $z$, or $m^2$, enabling the network to learn specialized representations for complementary aspects of QCD jet dynamics simultaneously.
- **Grad-CAM explainability for multi-graph GNNs**: Angular separation ($\Delta$, 40.72%) and relative transverse momentum ($k_T$, 35.67%) together account for ~76% of classification decisions, consistent with soft-collinear factorization in perturbative QCD.
- **State-of-the-art classification performance** on the JetClass benchmark across 9 signal classes.
- **Generalization to real collider data**: Evaluated on the Aspen Open Jets dataset of real CMS collision data, demonstrating robust representations under detector effects, pileup, and reconstruction uncertainties.

---

## Physics-Motivated Interaction Features

For each pair of connected particles $(a, b)$ in a jet, we compute four kinematic observables capturing key aspects of jet substructure. Because these variables span many orders of magnitude, we use their logarithms — $(\ln \Delta,\ \ln k_T,\ \ln z,\ \ln m^2)$ — as edge features, consistent with the logarithmic measure arising from the QCD emission probability.

### Feature Definitions

| Feature | Formula | Physical Meaning |
|:-------:|:-------:|:----------------|
| $\Delta$ | $\Delta = \sqrt{(y_a - y_b)^2 + (\Delta\phi_{ab})^2}$ | Angular separation; encodes angular ordering and collinear emissions |
| $k_T$ | $k_T = \min(p_{T,a},\, p_{T,b}) \cdot \Delta$ | Relative transverse momentum; sets the scale for $\alpha_s(k_T)$, separates perturbative from non-perturbative regimes |
| $z$ | $z = \dfrac{\min(p_{T,a},\, p_{T,b})}{p_{T,a} + p_{T,b}}$ | Momentum fraction; quantifies energy sharing via the DGLAP splitting functions $P(z)$ |
| $m^2$ | $m^2 = (E_a + E_b)^2 - \lVert \mathbf{p}_a + \mathbf{p}_b \rVert^2$ | Lorentz-invariant mass squared; provides mass-scale sensitivity essential for heavy-flavor jet identification |

### Symbol Reference

| Symbol | Definition |
|:------:|:-----------|
| $y_i$ | Rapidity: $y_i = \frac{1}{2}\ln\frac{E_i + p_{z,i}}{E_i - p_{z,i}}$ |
| $\phi_i$ | Azimuthal angle of particle $i$ |
| $\Delta\phi_{ab}$ | Azimuthal angle difference wrapped to $[-\pi, \pi]$ |
| $p_{T,i}$ | Transverse momentum: $p_{T,i} = \sqrt{p_{x,i}^2 + p_{y,i}^2}$ |
| $\mathbf{p}_i$ | Three-momentum of particle $i$: $(p_{x,i}, p_{y,i}, p_{z,i})$ |
| $E_i$ | Energy of particle $i$ |

### QCD Motivation

The emission probability in perturbative QCD factorizes as:

$$dP \propto \alpha_s(k_T)\, \frac{dk_T}{k_T}\, \frac{d\Delta}{\Delta}\, P(z)\, dz$$

This factorization makes ($\Delta$, $k_T$) the natural axes of the Lund jet plane, with $z$ appearing in the full emission probability and $m^2$ providing additional sensitivity to heavy-quark mass effects (dead-cone suppression). In the soft-collinear limit, this reduces to uniform emission density in $d(\ln k_T)\, d(\ln \Delta)$, making the logarithmic forms of these variables the natural inputs for a QCD-informed network.

> Dreyer, Salam & Soyez (2018). *The Lund Jet Plane.* [arXiv:1807.04758](https://arxiv.org/abs/1807.04758)
> Dreyer & Qu (2021). *Jet tagging in the Lund plane with graph networks.* [arXiv:2012.08526](https://arxiv.org/abs/2012.08526)

---

## Architecture

E-PCN processes each jet through four parallel GNN branches — one per kinematic variable — each consisting of alternating **Chebyshev graph convolutions (ChebConv)** and **edge convolutions (EdgeConv)** (ChebConv → EdgeConv → ChebConv → EdgeConv → ChebConv). Each branch produces a 64-dimensional jet-level embedding via mean pooling. The four embeddings are stacked into a 4×64 matrix and combined by a **1×1 convolution**, which learns to weight the kinematic representations adaptively. Two fully connected layers with dropout (rate 0.1) produce the final class probabilities via softmax.

### Explainability via Grad-CAM

We adapt Grad-CAM to the multi-graph setting by computing, for each graph branch, the product of gradient magnitude and embedding magnitude averaged over the 64 embedding dimensions. This yields a scalar importance score per kinematic variable, normalized to percentages summing to 100%.

**Global feature importance (averaged over all jet classes):**

| Variable | Importance | Role |
|:--------:|:----------:|:-----|
| $\Delta$ (angular separation) | **40.72%** | Dominant; encodes collinear structure |
| $k_T$ (transverse momentum) | **35.67%** | Strong; encodes soft radiation scale |
| $z$ (momentum fraction) | 14.06% | Moderate; encodes energy splitting |
| $m^2$ (invariant mass) | 9.54% | Lowest global; elevated for heavy flavor |

The 76% combined importance of $\Delta$ and $k_T$ is consistent with the soft-collinear factorization structure of perturbative QCD. Class-specific variations match established QCD mechanisms: enhanced $k_T$ sensitivity for gluon jets (Casimir scaling), elevated $\Delta$ for leptonic channels with missing energy, and increased $m^2$ for bottom-quark jets (dead-cone effect).

---

## Datasets

### JetClass
A large-scale benchmark comprising 100M jets across 10 classes (9 signal + 1 background), generated with Pythia 8.230. Signal classes include Higgs boson decays ($H \to b\bar{b}$, $H \to c\bar{c}$, $H \to gg$, $H \to 4q$, $H \to \ell\nu qq'$), top quark decays ($t \to bqq'$, $t \to b\ell\nu$), and electroweak boson decays ($W \to qq'$, $Z \to q\bar{q}$). We train on 1M jets (100K per class) and evaluate on the full 20M jet test set.

### Aspen Open Jets
Approximately **178 million high-p<sub>T</sub> jets** from the **CMS 2016 JetHT proton-proton collision Open Data** are used.

Since ground-truth class labels are not publicly available, representation quality is assessed using **unsupervised clustering metrics**, including the **Davies-Bouldin Index (DBI)** and **Dunn Index**, after training with the **DeepCluster** algorithm.

---

## Getting Started

### Installation

```bash
git clone https://github.com/ccdsiub/E-PCN.git
cd E-PCN
pip install -r requirements.txt
```

### Training Configuration

Key hyperparameters from the paper:

| Parameter | Value |
|:----------|:-----:|
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| LR Scheduler | OneCycleLR |
| Batch Size | 256 |
| Hidden Dimension | 64 |
| Graph Branches | 4 |
| k-NN neighbors | 3 |
| Conv. Layers | 5 (per branch) |
| Dropout Rate | 0.1 |
| Max Epochs | 500 |
| Early Stop patience | 10 epochs |

### Repository Structure

```
E-PCN/
├── raqib-pcn-experiments/                              # Main experiment scripts and notebooks
├── pythia-data-gen.md                                  # Data generation tutorial for Pythia
├── pythia-installation.md                              # Pythia installation guide
├── pythia-jet-tagging-data-generation-tutorial.md      # Jet tagging data generation walkthrough
├── pythia-python-guide.md                              # Python interface guide for Pythia
├── requirements.txt                                    # Python dependencies
└── README.md                                           # This file
```

---

## References

| # | Title | Venue |
|:-:|:------|:-----:|
| 1 | E-PCN: Jet Tagging with Explainable Particle Chebyshev Networks | [arXiv:2512.07420](https://arxiv.org/abs/2512.07420) |
| 2 | PCN: A Deep Learning Approach to Jet Tagging Using Chebyshev Graph Convolutions | [JHEP 2024](https://link.springer.com/article/10.1007/JHEP07(2024)247) |
| 3 | The Lund Jet Plane | [JHEP 2018](https://link.springer.com/article/10.1007/JHEP12(2018)064) |
| 4 | Jet Tagging in the Lund Plane with Graph Networks | [JHEP 2021](https://link.springer.com/article/10.1007/JHEP03(2021)052) |
| 5 | Particle Transformer for Jet Tagging (ParT) | [ICML 2022](https://proceedings.mlr.press/v162/qu22b.html) |
| 6 | ParticleNet: Jet Tagging via Particle Clouds | [Phys. Rev. D 2020](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.056019) |
| 7 | JetClass Dataset | [Zenodo](https://zenodo.org/record/6619768) |
| 8 | Aspen Open Jets | [ML: Sci. Tech. 2025](https://iopscience.iop.org/article/10.1088/2632-2153/adafc8) |
| 9 | Grad-CAM | [ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html) |

---

## Acknowledgements

We thank the [CERN Open Data Portal](http://opendata.cern.ch/) for providing high-quality collision data, and the original PCN authors for the base architecture. This research is partially supported by research grants from Independent University, Bangladesh (IUB).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use this work, please cite:

```bibtex
@article{islam2025epcn,
  title     = {E-PCN: Jet Tagging with Explainable Particle Chebyshev Networks Using Kinematic Features},
  author    = {Islam, Md Raqibul and Khan, Adrita and Hossain, Mir Sazzat and Siddiqui, Choudhury Ben Yamin and Hossain, Md. Zakir and Khan, Tanjib and Momen, M. Arshad and Ali, Amin Ahsan and Rahman, AKM Mahbubur},
  journal   = {arXiv preprint arXiv:2512.07420},
  year      = {2025}
}
```

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
  <br><br>
  <a href="mailto:sazzat@iub.edu.bd"><img src="https://img.shields.io/badge/Mir%20Sazzat%20Hossain-Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email Mir Sazzat Hossain"/></a>
</p>
