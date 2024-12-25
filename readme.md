# ECVA

## Overview

We develop **ECVA**, a new benchmark for causation understanding of video anomaly. ECVA is the first large-scale benchmark focused on the causation of video anomalies. Compared with existing datasets, our dataset is more comprehensive and more challenging with much higher-quality annotations. This work is an extension of **"Uncovering What, Why and How: A Comprehensive Benchmark for Causation Understanding of Video Anomaly" [CVPR2024]**

 [**Uncovering What, Why and How: A Comprehensive Benchmark for Causation Understanding of Video Anomaly**](https://github.com/fesvhtr/CUVA) 

[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/fesvhtr/CUVA)  [![github](https://img.shields.io/github/stars/fesvhtr/CUVA.svg?style=social)](https://github.com/fesvhtr/CUVA) [![arXiv](https://img.shields.io/badge/Arxiv-2412.07183-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.07183) [![dataset](https://img.shields.io/badge/Dataset-modelscope-green)](https://www.modelscope.cn/datasets/gouchenyi/ECVA)
---

## Table of Contents
- [Installation](#installation)
- [Benchmark and Evaluation Metric](#Benchmark)
- [Train Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
- [Inference](#inference)
- [Acknowledgement](#Acknowledgement)
- [License](#license)

---

## Installation

To install and set up the environment, follow these steps:

```bash
git clone git@github.com:Dulpy/ECVA.git
cd multimodal-video-large-model
pip install -r requirements.txt
```

## Benchmark and Evaluation Metric
- We develop **ECVA**, a new benchmark for causation understanding of video anomaly. ECVA is the first large-scale benchmark focused on the causation of video anomalies. Compared with existing datasets, our dataset is more comprehensive and more challenging with much higher-quality annotation.

- Our ECVA dataset contains 2240 video clips and 6720 question-answer pairs, the total length of these videos is 88.16 hours, and the average frames of videos is 8460. The frames are extracted from the original videos at a rate of 60 FPS. The videos encompass a wide range of domain

- You can download the original video data from this link: [Download Original Video Data](your_link_address)

- The ECVA video data along its annotation can be found in https://www.modelscope.cn/datasets/gouchenyi/ECVA/files

![Classes](assert/classes-new_00.png)




### The proposed evaluation mertic
The proposed evaluation mertic mainly measures the performance of the model comprehensively through the following three aspects.

- #### Basic Reasoning

  **In the Basic Reasoning part, we use the GPT model to assess whether the candidate answers comprehensively cover all key phrases and rate the answers based on their logical coherence.** 

- #### Consistency

  **For the Consistency evaluation, we leverage the binarity of the GPT to score the candidate answers.**

- #### Hallucination

  **As for the Hallucination part, we remove key frames from the video and input it into the VLMs to observe how consistent the model's responses are with or without the key frames.**


![Evaluation_metric](assert/Evaluation_metric_00.png)

### Evaluate your results on ECVA

#### 1. Reformat your results

For each video response, you need to organize it into the following format:
```bash
[{
  "video_file": '00001.mp4'
  "prompt": 'Give a detailed description of the anomalous segment in the video. Please remember to describe the details of the incident'
  "output": 'your model's response to this prompt'
  "task_type": 'Description'
  "human_expert_answer": 'The standard answer for the task'
},
]
```

#### 2. Evaluate your results

Prepare the model's answers and our benchmark answers, then use the script [here](AnomEval/evaluating_system_v2) to score them with GPT assistant. Because GPT will be used to assist in the evaluation, you will need to fill in your own key in the [relevant configuration file](AnomEval/evaluating_system_v2/config.py)

### Evaluate your results on traditional mertic

#### 1. Reformat your results as shown above

#### 2. Evaluate your results

Prepare the model's answers and our benchmark answers, then use the script [here](eval_traditional) to evaluate them use **[BLUE](https://github.com/neural-dialogue-metrics/BLEU), [ROUGE](https://github.com/pltrdy/rouge), [BLEURT](https://github.com/google-research/bleurt) and [UNIEVAL](https://github.com/maszhongming/UniEval)**.

## Training Dataset Preparation

We introduce a novel video large language model named **Anomaly Shield**  (AnomShield), which is designed to address the three challenges presented by ECVA. You can re-organize the annotated video/image sft data according to the following format and place the image/video data in the path **ECVA/datasets/pretraining/** and **ECVA/datasets/videosft/**

```bash
[
    {
        "id": 0,
        "video": "images/xxx.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat are the colors of the bus in the image?"
            },
            {
                "from": "gpt",
                "value": "The bus in the image is white and red."
            },
            ...
        ],
    }
    {
        "id": 1,
        "video": "videos/xxx.mp4",
        "conversations": [
            {
                "from": "human",
                "value": "<video>\nWhat are the main activities that take place in the video?"
            },
            {
                "from": "gpt",
                "value": "The main activities that take place in the video are the preparation of camera equipment by a man, a group of men riding a helicopter, and a man sailing a boat through the water."
            },
            ...
        ],
    },
    ...
]
```

## Model Training

### 1. Prepare CLIP and Mistral Weight

 - For Vision-Encoder, similar to most multi-modal large models, AnomShield uses the CLIP series as the visual encoder. You can download the related pre-trained weights from [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14).
   
 - For the base model, we utilize the powerful Mistral series to help analyze the video content and provide reliable, accurate answers. You can download the related pre-trained weights from [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2).


### 2. Pretrain Command
```bash
cd ECVA/scripts/vllava/mamba/
./pretrain.sh
```
### 3. Video SFT Command
```bash
cd ECVA/scripts/vllava/mamba/
./finetune.sh
```

## Inference

Video/Image Inference. We have inherited the inference code from **VideoLLaMA2**.You can refer to the **inference.ipynb** to implement the model inference, and you need to prepare the relevant model weights according to the instructions in the script.

```bash

cd ECVA/
run inference.ipynb on the jupyter environment

```

## Acknowledgement
The codebase of **ECVA** is adapted from [**VideoLLaMA2**](https://github.com/DAMO-NLP-SG/VideoLLaMA2). We are grateful for the foundational work done by the VideoLLaMA2 team, which has significantly contributed to the development of this project.

## License

## Cite
If you find our work useful for your research, please consider citing.

```bash

@article{du2024exploring,
  title={Exploring What Why and How: A Multifaceted Benchmark for Causation Understanding of Video Anomaly},
  author={Du, Hang and Nan, Guoshun and Qian, Jiawen and Wu, Wangchenhui and Deng, Wendi and Mu, Hanqing and Chen, Zhenyan and Mao, Pengxuan and Tao, Xiaofeng and Liu, Jun},
  journal={arXiv preprint arXiv:2412.07183},
  year={2024}
}

```
