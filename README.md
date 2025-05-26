<h1 align="center">CAT-LVDM: Corruption-Aware Training of Latent Video Diffusion Models</h1>

<p align="center">
  <a href="https://github.com/chikap421/catlvdm">
    <img src="https://img.shields.io/badge/Project-Page-green?style=flat-square&logo=github">
  </a>
  <a href="https://arxiv.org/abs/2405.12345">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square">
  </a>
  <a href="https://huggingface.co/Chikap421/catlvdm-checkpoints/tree/main">
    <img src="https://img.shields.io/badge/Model-HuggingFace-blue?style=flat-square">
  </a>
  <a href="https://colab.research.google.com/github/catlvdm/demo/blob/main/notebook.ipynb">
    <img src="https://img.shields.io/badge/Demo-Colab-orange?style=flat-square">
  </a>
</p>

<p align="center">
  <i>This repository contains the code for CAT-LVDM: a corruption-aware training framework for robust latent video diffusion models.</i>
</p>

---

<p align="center">
  <img src="assets/overview.png" width="700"/>
</p>

<p align="center"><b>Figure:</b> <i>(a) Visual comparison of generation quality across corruption schemes</i> (BCNI, Gaussian, Uniform, Clean) for the prompt <b>"Cat plays with holiday baubles."</b> <i>(b) Quantitative summary</i> of performance on FVD (↓), VBench (↑), and EvalCrafter (↑). Our method, <b>BCNI (ours)</b>, outperforms others in both semantic fidelity and motion realism under structured noise.</p>

---

### Robustness under Corruption

<p align="center"><b>Prompt:</b> Rotation, close-up, falling drops of water on ripe cucumbers.</p>

<table align="center">
<tr>
  <td align="center"><img src="assets/bcni/1.gif" width="180px"><br><b>BCNI (ours)</b></td>
  <td align="center"><img src="assets/gaussian/1.gif" width="180px"><br><b>Gaussian</b></td>
  <td align="center"><img src="assets/uniform/1.gif" width="180px"><br><b>Uniform</b></td>
  <td align="center"><img src="assets/clean/1.gif" width="180px"><br><b>Clean</b></td>
</tr>
</table>

<p align="center"><b>Prompt:</b> Seascape of coral reef in caribbean sea.</p>

<table align="center">
<tr>
  <td align="center"><img src="assets/bcni/2.gif" width="180px"><br><b>BCNI (ours)</b></td>
  <td align="center"><img src="assets/gaussian/2.gif" width="180px"><br><b>Gaussian</b></td>
  <td align="center"><img src="assets/uniform/2.gif" width="180px"><br><b>Uniform</b></td>
  <td align="center"><img src="assets/clean/2.gif" width="180px"><br><b>Clean</b></td>
</tr>
</table>

<p align="center"><b>Prompt:</b> Walking with Dog.</p>

<table align="center">
<tr>
  <td align="center"><img src="assets/sacn/3.gif" width="180px"><br><b>SACN (ours)</b></td>
  <td align="center"><img src="assets/gaussian/3.gif" width="180px"><br><b>Gaussian</b></td>
  <td align="center"><img src="assets/uniform/3.gif" width="180px"><br><b>Uniform</b></td>
  <td align="center"><img src="assets/clean/3.gif" width="180px"><br><b>Clean</b></td>
</tr>
</table>

<p align="center"><b>Prompt:</b> Close up of indian biryani rice slowly cooked and stirred.</p>

<table align="center">
<tr>
  <td align="center"><img src="assets/bcni/4.gif" width="180px"><br><b>BCNI (ours)</b></td>
  <td align="center"><img src="assets/gaussian/4.gif" width="180px"><br><b>Gaussian</b></td>
  <td align="center"><img src="assets/uniform/4.gif" width="180px"><br><b>Uniform</b></td>
  <td align="center"><img src="assets/clean/4.gif" width="180px"><br><b>Clean</b></td>
</tr>
</table>

<p align="center"><b>Prompt:</b> Natural colorful waterfall.</p>

<table align="center">
<tr>
  <td align="center"><img src="assets/bcni/5.gif" width="180px"><br><b>BCNI (ours)</b></td>
  <td align="center"><img src="assets/gaussian/5.gif" width="180px"><br><b>Gaussian</b></td>
  <td align="center"><img src="assets/uniform/5.gif" width="180px"><br><b>Uniform</b></td>
  <td align="center"><img src="assets/clean/5.gif" width="180px"><br><b>Clean</b></td>
</tr>
</table>

<p align="center"><b>Prompt:</b> Two business women using a touchpad in the office are busy discussing matters.</p>

<table align="center">
<tr>
  <td align="center"><img src="assets/bcni/6.gif" width="180px"><br><b>BCNI (ours)</b></td>
  <td align="center"><img src="assets/gaussian/6.gif" width="180px"><br><b>Gaussian</b></td>
  <td align="center"><img src="assets/uniform/6.gif" width="180px"><br><b>Uniform</b></td>
  <td align="center"><img src="assets/clean/6.gif" width="180px"><br><b>Clean</b></td>
</tr>
</table>

<p align="center"><b>Prompt:</b> Technician in white coat walking down factory storage, opening laptop and starting work.</p>

<table align="center">
<tr>
  <td align="center"><img src="assets/bcni/7.gif" width="180px"><br><b>BCNI (ours)</b></td>
  <td align="center"><img src="assets/gaussian/7.gif" width="180px"><br><b>Gaussian</b></td>
  <td align="center"><img src="assets/uniform/7.gif" width="180px"><br><b>Uniform</b></td>
  <td align="center"><img src="assets/clean/7.gif" width="180px"><br><b>Clean</b></td>
</tr>
</table>







<!-- GETTING STARTED -->

### 1. Getting Started

This repo implements **CAT-LVDM**, a corruption-aware training framework that improves the robustness of latent video diffusion models via structured noise (BCNI/SACN).

#### Requirements

```
conda create -n catlvdm python=3.8
conda activate catlvdm
pip install -r requirements.txt
```

Ensure compatibility with `torch==2.1.2` compiled with `nvcc 12.1`.

### 2. Checkpoints

We provide a comprehensive suite of CAT-LVDM model checkpoints trained under diverse structured corruption settings across multiple noise levels (2.5%, 5%, 7.5%, 10%, 15%, 20%). These are hosted at:  
👉 [https://huggingface.co/Chikap421/catlvdm-checkpoints](https://huggingface.co/Chikap421/catlvdm-checkpoints)

To download the **base model (ModelScope)** and optionally the CAT-LVDM checkpoints, run:

➡️ [`models/download.sh`](models/download.sh)

This script installs Git LFS and clones the required base model. To also download CAT-LVDM checkpoints, simply uncomment the final line in the script.

---

#### 📊 Corruption Types in CAT-LVDM

CAT-LVDM introduces both **embedding-level** and **text-level** corruption methods to evaluate model robustness under structured noise. Each corruption scheme is applied across six corruption strengths (ρ = 2.5%, 5%, 7.5%, 10%, 15%, 20%).

Each folder on [Hugging Face](https://huggingface.co/Chikap421/catlvdm-checkpoints) follows the format: `corruptiontype_strength`, e.g., `bcni_10`, `swap_5`.
The folder `results_2M_train` is used to denote the clean (non-corrupted) training setup without any embedding or text-level noise.

---

##### 🧬 Embedding-Level Corruptions

| Folder Prefix | Corruption Type                     | Description |
|---------------|-------------------------------------|-------------|
| `bcni`        | Batch-Centered Noise Injection      | Perturbs embeddings along intra-batch semantic axes. Encourages temporal coherence and semantic preservation. |
| `sacn`        | Spectrum-Aware Contextual Noise     | Injects spectral noise aligned with principal low-frequency components. |
| `gaussian`    | Isotropic Gaussian Noise            | Adds unstructured Gaussian noise to each dimension. |
| `uniform`     | Isotropic Uniform Noise             | Injects bounded uniform noise independently across dimensions. |
| `tani`        | Temporally-Aligned Noise Injection  | Aligns noise with motion direction across adjacent video frames. |
| `hscan`       | Hierarchical Spectral Corruption    | Applies multiscale spectral noise with SACN + Gaussian fusion. |

---

##### ✏️ Text-Level Corruptions

| Folder Prefix | Corruption Type      | Description |
|---------------|----------------------|-------------|
| `add`         | Text Addition        | Randomly inserts new tokens into the prompt. |
| `remove`      | Text Removal         | Deletes tokens from the input text. |
| `replace`     | Text Replacement     | Replaces existing tokens with others sampled from batch. |
| `swap`        | Text Swap            | Swaps positions of two tokens in the sequence. |
| `perturb`     | Text Perturbation    | Replaces tokens with visually or semantically noisy variants. |




### 3. Inference

To run inference with pre-trained CAT-LVDM checkpoints:

```bash
bash scripts/inference_deepspeed.sh
```

> Output videos are saved in `log_dir` (specify path in config).

Prompts should be formatted as a CSV file:
```
id,prompt
1,A scientist works in a clean lab.
2,A camel walks across the desert.
```
We provide curated sample prompts in: [`prompts/sampled_captions.json`](prompts/sampled_captions.json)

Configurable options are defined in [`configs/t2v_inference_deepspeed.yaml`](configs/t2v_inference_deepspeed.yaml).


### 4. Training

#### Dataset Setup

This repository supports training on the WebVid-2M training split, and inference on the WebVid-2M validation split, MSR-VTT, MSVD, and UCF101 datasets.

#### Training Command

```bash
bash scripts/train_deepspeed.sh
```

### 5. Multi-Corruption Parallel Training & Inference

To efficiently run parallel training or inference experiments across multiple corruption settings (e.g., BCNI, SACN), ablation variants, or noise levels, refer to the multi-script setup below:

#### Parallel Training
Use the following script to launch multi-GPU training across various corruption settings:
```bash
bash scripts/multi_train.sh
```

#### Parallel Inference
Similarly, run multi-setting inference using:
```bash
bash scripts/multi_inference.sh
```

These scripts automatically adjust corruption schemes and noise parameters defined in:
- `configs/t2v_train_deepspeed.yaml` (for training)
- `configs/t2v_inference_deepspeed.yaml` (for inference)

#### TensorBoard

```bash
tensorboard --logdir=tensorboard_log/catlvdm
```

> Logs are saved in `tensorboard_log/catlvdm/`.








<!-- ROADMAP -->
## TODO
- [x] Structured corruption injection
- [x] Full benchmark on 4 datasets
- [x] Model checkpoints release
- [ ] arXiv upload
- [ ] Collab Demo
- [ ] Model evaluation







<!-- LICENSE -->
## License

Distributed under the MIT License. See [LICENSE.txt](./LICENSE.txt) for more information.






<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Our implementation is adapted from [DEMO](https://github.com/pr-ryan/DEMO) and [VGen](https://github.com/ali-vilab/VGen). We thank the authors for their open-source contributions.
