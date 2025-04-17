# **Volterra Neural Networks (VNNs) with RKHS Projections**  
**Final Project - Deep Learning (MVA 2024)**  

This repository contains our final project for the Deep Learning course of the MVA program. The goal of this project is to **reproduce the results from the paper "Conquering the CNN Over-Parameterization Dilemma: A Volterra Filtering Approach for Action Recognition"** ([Roheda & Krim, AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6870)), which introduces **Volterra Neural Networks (VNNs)** as an alternative to standard CNNs for action recognition.  

In addition to reproducing the original results, we aim to **improve the VNN model by leveraging Reproducing Kernel Hilbert Space (RKHS) projections**. The RKHS formulation allows us to:  
✔️ Reduce the computational cost of high-order Volterra expansions.  
✔️ Improve generalization through better function space regularization.  
✔️ Maintain interpretability while increasing expressive power.  

---

## **Project Objectives**  
1️⃣ **Reproduce the original VNN results** on action recognition datasets (e.g., HMDB-51, UCF-101).  
2️⃣ **Analyze the strengths and limitations** of the Volterra formulation.  
3️⃣ **Propose an RKHS-based extension** to improve efficiency and generalization.  

We hypothesize that RKHS projections can **reduce the number of required parameters while preserving expressivity**, making the model more computationally efficient compared to the original VNN formulation.  

---

## **Key Features**  
✅ **Implementation of Volterra Neural Networks** – Reproduction of the baseline model.  
✅ **RKHS-based Projections** – Alternative high-order interactions using kernel approximations.  
✅ **Flexible Model Configurations** – Support for RGB, optical flow, and multi-modal inputs.  
✅ **Efficient Training & Inference** – Precomputed features for reduced training time.  

---

## **Getting Started**  

### **Prerequisites**  
Ensure you have **Python 3.x** and install dependencies using:  

```bash
pip install -r requirements.txt
```

### **Training the Baseline VNN Model**  

1️⃣ **Set Dataset Path** – Modify `mypath.py` with dataset locations.  
2️⃣ **Choose Model Configuration** – Adjust hyperparameters in `train_VNN_fusion_highQ.py`.  
3️⃣ **Run Training**  
```bash
python3 train_VNN_fusion_highQ.py
```  
4️⃣ **(Optional) Precompute Optical Flow** – Reduces training time by caching intermediate representations.  

---

## **RKHS-Based Improvement Strategy**  
The original Volterra Neural Networks (VNNs) model nonlinear interactions using **Volterra series expansions**, which can be computationally expensive for higher-order terms.  

We propose an alternative approach:  
🔹 **Reformulate high-order interactions using RKHS projections**.  
🔹 **Use kernel approximations to reduce complexity**.  
🔹 **Introduce functional regularization to improve generalization**.  

The RKHS-based implementation can be found in `networks/vnn_rkhs.py`.  

---

## **Project Structure**  
```
volterra/
├── config/                # Configuration files
├── data/                  # Datasets and preprocessing scripts
├── jobs/                  # SLURM batch scripts for cluster execution
├── logs/                  # Training and evaluation logs
├── models/                # Trained models and checkpoints
├── networks/              # Model architectures
│   ├── vnn_rgb_of_complex.py  # Standard VNN architecture
│   ├── vnn_rkhs.py            # RKHS-based Volterra implementation
├── inference.py           # Script for model inference
├── requirements.txt       # Dependencies list
├── train_VNN_fusion_highQ.py  # Main training script
└── README.md              # Project documentation
```

---

## **Results & Findings**  
📊 **Performance Evaluation**  
We compare the **baseline VNN model** across the HMDB51 datasets.  

---

## **Future Work**  
🚀 Exploring different kernel choices for RKHS embedding.  
🚀 Extending to **self-supervised learning** and **few-shot learning**.  
🚀 Investigating hybrid **VNN-CNN architectures** for large-scale applications.  

---

## **Citation**  
If you use our work, please cite the original paper:  

```bibtex
@inproceedings{roheda2020conquering,
  title={Conquering the cnn over-parameterization dilemma: A volterra filtering approach for action recognition},
  author={Roheda, Siddharth and Krim, Hamid},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={07},
  pages={11948--11956},
  year={2020}
}
```

## Contact  

For questions or feedback, feel free to reach out:  
👨‍💻 **Authors:**: Clément Leprêtre & Ilyess Doragh
🏫 **Institution:** CentraleSupélec – MVA Deep Learning Course  
📩 **Contact:**: clement.lepretre@student-cs.fr / ilyess.doragh@student-cs.fr

---

## **Acknowledgements**  
We thank **CentraleSupélec** and the **MVA faculty** for their guidance, as well as **Siddharth Roheda & Hamid Krim** for their foundational work on Volterra Neural Networks.  
