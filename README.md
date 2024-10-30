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



<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">DEMO: Enhancing Motion in Text-to-Video Generation with Decomposed Encoding and Conditioning</h3>

  <h4 align="center">Penghui Ruan, Pichao Wang, Divya Saxena, Jiannong Cao, Yuhui Shi</h4>
  <h4 align="center">Accepted by NeurIPS 2024 Poster</h4>

</div>

![](assets/architecture.jpeg)



<div style="display: flex; text-align: center; gap: 5px; ">

  <figure style="width: 25%; margin: 0;">
      <figcaption>Lavie</figcaption>
    <img src="assets/lavie/1.gif" alt="Lavie GIF" width="100%">

  </figure>

  <figure style="width: 25%; margin: 0;">
      <figcaption>VideoCrafter2</figcaption>
    <img src="assets/videocrafter2/1.gif" alt="VideoCrafter2 GIF" width="100%">

  </figure>

  <figure style="width: 25%; margin: 0;">
      <figcaption>ModelScope (Base Model)</figcaption>
    <img src="assets/modelscope/1.gif" alt="ModelScope GIF" width="100%">

  </figure>

  <figure style="width: 25%; margin: 0;">
    <figcaption>Demo</figcaption>
    <img src="assets/demo/1.gif" alt="Demo GIF" width="100%">
  </figure>
</div>
<h3 align="center"> Slow motion flower petals fall from a blossom, landing softly on the ground.</h3>


<div style="display: flex; text-align: center; gap: 5px; ">

  <figure style="width: 25%; margin: 0;">
    <img src="assets/lavie/2.gif" alt="Lavie GIF" width="100%">
  </figure>

  <figure style="width: 25%; margin: 0;">
    <img src="assets/videocrafter2/2.gif" alt="VideoCrafter2 GIF" width="100%">
  </figure>

  <figure style="width: 25%; margin: 0;">
    <img src="assets/modelscope/2.gif" alt="ModelScope GIF" width="100%">
  </figure>

  <figure style="width: 25%; margin: 0;">
    <img src="assets/demo/2.gif" alt="Demo GIF" width="100%">
  </figure>
</div>
<h3 align="center">An old man with white hair is shown speaking.</h3>


<div style="display: flex; text-align: center; gap: 5px; ">

  <figure style="width: 25%; margin: 0;">
    <img src="assets/lavie/3.gif" alt="Lavie GIF" width="100%">

  </figure>

  <figure style="width: 25%; margin: 0;">
    <img src="assets/videocrafter2/3.gif" alt="VideoCrafter2 GIF" width="100%">

  </figure>

  <figure style="width: 25%; margin: 0;">
    <img src="assets/modelscope/3.gif" alt="ModelScope GIF" width="100%">

  </figure>

  <figure style="width: 25%; margin: 0;">
    <img src="assets/demo/3.gif" alt="Demo GIF" width="100%">
  </figure>
</div>
<h3 align="center">Jockeys racing.</h3>





<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
#### Install ffmpeg
```bash
sudo apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
```
#### Environment Preparation
  ```bash
  git clone git@github.com:PR-Ryan/DEMO.git
  ```

  ```bash
  conda create -n demo python=3.8
  conda activate demo
  pip install -r requirements.txt
  ```



### Inference
```bash
bash scripts/inference_deeepspeed.sh
```

#### Download Pretrained Model
```bash
bash models/download.sh
```

#### Download our fine-tuned [checkpoints](https://huggingface.co/Ryan-PR/DEMO) from huggingface.

#### Prepare inference prompt in prompts/your_prompt.csv. Example prompt file as:
```bash
id,prompt
1,a fat dog is playing in the yard.
2,a fat car is parked by the road.
3,a fat balloon is floating in the air.
```


### Training

#### Dataset Preparation
Follow the instruction and download [Web-Vid](https://github.com/m-bain/webvid) dataset. If you prefer to use your own dataset, please refer to tools/datasets/video_datasets.py to define your own dataset and preprocessing step.




#### Download Pretrained models

```bash
bash models/download.sh
```

#### Train the model
```bash
bash scripts/train_deeepspeed.sh
```








<!-- ROADMAP -->
## Roadmap

- [x] Open source model weights.
- [x] Open source inference and training code.
- [ ] Huggingface demo.
- [ ] gradio application.


<p align="right">(<a href="#readme-top">back to top</a>)</p>





<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Penghui Ruan - penghui.ruan@connect.polyu.hk

Project Link: [https://pr-ryan.github.io/DEMO-project/](https://pr-ryan.github.io/DEMO-project/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This repo is heavily built upon [VGen](https://github.com/ali-vilab/VGen) from alibaba. We sincerely thanks for their effort to contribting the open-source conmmunity.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


