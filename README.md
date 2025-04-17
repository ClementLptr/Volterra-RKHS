# Volterra Neural Networks (VNNs) with RKHS Projections

**Research Project – Mention MDS, CentraleSupélec (2024-2025)**

This repository contains the research project conducted as part of the **MDS (Mathématiques, Données, Sciences)** program at CentraleSupélec. The project focuses on **reproducing and extending the results from the paper "Volterra Neural Networks (VNNs)"** ([Roheda et al., JMLR 2024](https://www.jmlr.org/papers/v25/21-1130.html)), which introduces **Volterra Neural Networks (VNNs)** as a framework for modeling nonlinear interactions in data, offering an alternative to traditional convolutional neural networks (CNNs) for tasks such as action recognition.

We aim to **reproduce the original VNN results and enhance the model by implementing Reproducing Kernel Hilbert Space (RKHS) projections on a one-stream (RGB) VNN architecture**. This approach is compared to the baseline VNN model presented in the original paper to evaluate improvements in efficiency and performance.

> **Note**: This README contains LaTeX equations for mathematical formulations. To view them correctly, use a Markdown viewer that supports MathJax (e.g., GitHub, VS Code with a LaTeX plugin, or Jupyter Notebook).

## Project Objectives

1. **Reproduce the original VNN results** on action recognition datasets (e.g., HMDB-51, UCF-101) using the baseline model.
2. **Implement RKHS projections on a one-stream (RGB) VNN** to model nonlinear interactions more efficiently.
3. **Compare the RKHS-enhanced VNN** with the baseline VNN from Roheda et al. (JMLR 2024) to assess improvements in computational efficiency and generalization.

We hypothesize that RKHS projections applied to a one-stream (RGB) VNN can **reduce the number of required parameters while preserving expressivity**, making the model more computationally efficient compared to the original VNN formulation.

## Key Features

- **Implementation of Volterra Neural Networks**: Reproduction of the baseline model.
- **RKHS-based Projections**: Alternative high-order interactions using kernel approximations for one-stream (RGB) inputs.
- **One-Stream RGB Architecture**: Focused implementation for efficient action recognition.
- **Efficient Training & Inference**: Precomputed features for reduced training time.

## Getting Started

### Prerequisites

Ensure you have **Python 3.x** installed and install dependencies using:

```bash
pip install -r requirements.txt
```

### Training the Baseline VNN Model

1. **Set Dataset Path**: Modify `mypath.py` with dataset locations.
2. **Choose Model Configuration**: Adjust hyperparameters in `train_VNN_fusion_highQ.py`.
3. **Run Training**:

```bash
python3 train_VNN_fusion_highQ.py
```

4. **(Optional) Precompute Features**: Reduces training time by caching intermediate representations.

### Training the RKHS-Enhanced VNN Model

The RKHS-based implementation for the one-stream (RGB) VNN is located in `networks/vnn_rkhs.py`. To train this model:

1. Configure `train_VNN_fusion_highQ.py` to use the RKHS model.
2. Run:

```bash
python3 train_VNN_fusion_highQ.py --model vnn_rkhs
```

## RKHS-Based Improvement Strategy

The original Volterra Neural Networks (VNNs) model nonlinear interactions using **Volterra series expansions**, which can be computationally expensive for higher-order terms. Our approach focuses on a **one-stream (RGB) VNN architecture**, where we:

- **Reformulate high-order interactions using RKHS projections** to reduce computational complexity.
- **Use kernel approximations** to efficiently model nonlinear relationships in RGB data.
- **Introduce functional regularization** to improve generalization.

By applying RKHS projections to a one-stream VNN, we aim to achieve comparable or better performance than the baseline VNN model from Roheda et al. (JMLR 2024) with lower computational cost. The RKHS-based implementation can be found in `networks/vnn_rkhs.py`.

## Mathematical Foundations of RKHS Approximation

RKHS representations enable an implicit mapping of input data into a high-dimensional feature space where nonlinear relationships can be effectively modeled using linear methods. The connection between Volterra series and RKHS representations was formalized by Franz and Schölkopf (2006), who demonstrated that polynomial kernel regression provides a unifying framework for Wiener and Volterra theory.

Specifically, given an input-output relationship expressed through a Volterra series expansion:

$$
\sum_{k=0}^{n} W^k(x) = \eta^{(n)} \varphi^{(n)}(x)
$$

it is possible to approximate this expansion using an inhomogeneous polynomial kernel:

$$
k(x, y) = (\langle x, y \rangle + 1)^n
$$

where \( n \) is the order of the Volterra series, which naturally encodes higher-order interactions (Franz and Schölkopf, 2006).

For our one-stream (RGB) VNN, we focus on a second-order Volterra series, rewritten using the following RKHS projection:

<div align="center">
  <img src="figures/rkhs_formule.png" alt="RKHS Formula">
</div>

This formulation allows us to capture complex nonlinear interactions in RGB data while leveraging the computational efficiency of kernel methods, enabling a direct comparison with the baseline VNN model.

## Project Structure

```
volterra/
├── config/                # Configuration files
├── data/                  # Datasets and preprocessing scripts
├── jobs/                  # SLURM batch scripts for cluster execution
├── logs/                  # Training and evaluation logs
├── models/                # Trained models and checkpoints
├── networks/              # Model architectures
│   ├── vnn_rgb_of_complex.py  # Standard VNN architecture
│   ├── vnn_rkhs.py            # RKHS-based Volterra implementation for one-stream RGB
├── inference.py           # Script for model inference
├── requirements.txt       # Dependencies list
├── train_VNN_fusion_highQ.py  # Main training script
└── README.md              # Project documentation
```

## Results & Findings

**Performance Evaluation**  
We compare the **baseline VNN model** and the **RKHS-enhanced one-stream (RGB) VNN** across the HMDB51 datasets to evaluate improvements in accuracy and computational efficiency.

## Future Work

- Exploring different kernel choices for RKHS embedding in the one-stream architecture.
- Extending to **self-supervised learning** and **few-shot learning** for RGB inputs.
- Investigating hybrid **VNN-CNN architectures** for large-scale applications.

## Citation

If you use our work, please cite the original paper:

```bibtex
@article{roheda2024volterra,
  title={Volterra Neural Networks (VNNs)},
  author={Roheda, Siddharth and Krim, Hamid and Jiang, Bo},
  journal={Journal of Machine Learning Research},
  volume={25},
  pages={1--29},
  year={2024},
  url={https://www.jmlr.org/papers/v25/21-1130.html}
}
```

Additionally, for the mathematical foundations of the RKHS approximation, please cite:

```bibtex
@article{Franz2006,
  title={Reproducing kernel Hilbert spaces and Volterra series},
  author={Franz, M. and Sch{\"o}lkopf, B.},
  journal={Technical Report},
  year={2006}
}
```

## Contact

For questions or feedback, feel free to reach out:  
**Authors**: Clément Leprêtre & Salim Gazzeh  
**Institution**: CentraleSupélec – Mention MDS  
**Contact**: clement.lepretre@centralesupelec.fr / salim.gazzeh@centralesupelec.fr

## Acknowledgements

We express our gratitude to **CentraleSupélec** and the **MDS faculty** for their support. Special thanks to our supervisors, **JC Pesquet** and **H. Krim**, for their guidance, and to our tutor, **A. Minasyan**, for their valuable feedback. We also acknowledge **Siddharth Roheda**, **Hamid Krim**, and **Bo Jiang** for their foundational work on Volterra Neural Networks, and **M. Franz** and **B. Schölkopf** for their contributions to the RKHS-Volterra connection.