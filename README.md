# VCM: Vision Concept Modeling with Adaptive Vision Token Compression via Instruction Fine-Tuning

The official repository for **"VCM: Vision Concept Modeling with Adaptive Vision Token Compression via Instruction Fine-Tuning"**.

<p align="center">
       🤗 <a href="#">VCM-7B (Coming Soon)</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="#">VCM-13B (Coming Soon)</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2504.19627">Paper</a>&nbsp&nbsp
</p>

## Introduction
**VCM (Vision Concept Modeling)** is a novel framework designed to enhance the efficiency of Large Multimodal Models (LMMs). By introducing adaptive vision token compression during the instruction fine-tuning stage, VCM dynamically identifies and preserves essential visual concepts while reducing redundant tokens. This approach significantly lowers computational overhead without compromising performance on downstream multimodal tasks.

---

## Installation and Setup

VCM is built upon the [LLaVA](https://github.com/haotian-liu/LLaVA) framework. To use VCM, please follow these steps:

1.  **Clone the official LLaVA repository:**
    ```bash
    git clone https://github.com/haotian-liu/LLaVA.git
    cd LLaVA
    ```

2.  **Install the environment:**
    Follow the original LLaVA installation instructions to set up your Python environment and dependencies.

3.  **Apply VCM Modifications:**
    Replace the original `llava_arch.py` file in the LLaVA source code with the one provided in this repository:
    ```bash
    cp path/to/vcm/llava_arch.py llava/model/llava_arch.py
    ```

---

## Training and Inference

Once you have replaced the architecture file, you can follow the standard LLaVA training and inference pipelines. VCM will automatically handle the adaptive token compression based on the Vision Concept Modeling logic during the forward pass.

Refer to the [LLaVA Documentation](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#train) for detailed commands on:
- Pre-training (Feature Alignment)
- Visual Instruction Tuning

---

## Evaluation

We utilize the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) toolkit for comprehensive benchmarking. 


---

## Citation

If you find VCM useful for your research, please cite our paper:

```bibtex
@article{vcm2025,
  title={VCM: Vision Concept Modeling with Adaptive Vision Token Compression via Instruction Fine-Tuning},
  author={Run Luo and Renke Shan and Longze Chen and Ziqiang Liu and Lu Wang and Min Yang and Xiaobo Xia},
  journal={arXiv preprint arXiv:2504.19627},
  year={2025}
}
