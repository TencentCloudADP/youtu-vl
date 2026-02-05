<div align="center">

# <img src="assets/youtu-vl-logo.png" alt="Youtu-VL Logo" height="100px">

[📃 License](LICENSE) • [💻 Code](https://github.com/TencentCloudADP/youtu-vl) • [📑 Technical Report](https://arxiv.org/abs/2601.19798) • [📊 Benchmarks](#benchmarks) • [🚀 Getting Started](#quickstart) • [🤗 Models](https://huggingface.co/collections/tencent/youtu)
</div>

## 🎯 Introduction

**Youtu-VL** is a lightweight yet robust Vision-Language Model (VLM) built on the Youtu-LLM with 4B parameters. It pioneers Vision-Language Unified Autoregressive Supervision (VLUAS), which markedly strengthens visual perception and multimodal understanding. This enables a standard VLM to perform vision-centric tasks without task-specific additions. Across benchmarks, Youtu-VL stands out for its versatility, achieving competitive results on both vision-centric and general multimodal tasks.


## ✨ Key Features

  - **Comprehensive Vision-Centric Capabilities**: The model demonstrates strong, broad proficiency across classic vision-centric tasks, delivering competitive performance in visual grounding, image classification, object detection, referring segmentation, semantic segmentation, depth estimation, object counting, and human pose estimation.

  - **Promising Performance with High Efficiency**: Despite its compact 4B-parameter architecture, the model achieves competitive results across a wide range of general multimodal tasks, including general visual question answering (VQA), multimodal reasoning and mathematics, optical character recognition (OCR), multi-image and real-world understanding, hallucination evaluation, and GUI agent tasks.

  <p align="center">
      <img src="assets/youtu-vl-overview.png" width="90%"/>
  <p>

## 🤗 Model Download

| Model Name  | Description | Download |
| ----------- | ----------- |-----------
| Youtu-VL-4B-Instruct | Visual language model of Youtu-LLM | 🤗 [Model](https://huggingface.co/tencent/Youtu-VL-4B-Instruct)|
| Youtu-VL-4B-Instruct-GGUF | Visual language model of Youtu-LLM, in GGUF format | 🤗 [Model](https://huggingface.co/tencent/Youtu-VL-4B-Instruct-GGUF)|

## 🧠 Model Architecture Highlights

  - **Vision–Language Unified Autoregressive Supervision (VLUAS)**: Youtu-VL is built on the VLUAS paradigm to mitigate the text-dominant optimization bias in conventional VLMs, where visual signals are treated as passive conditions and fine-grained details are often dropped. Rather than using vision features only as inputs, Youtu-VL expands the text lexicon into a unified multimodal vocabulary through a learned visual codebook, turning visual signals into autoregressive supervision targets. Jointly reconstructing visual tokens and text explicitly preserves dense visual information while strengthening multimodal semantic understanding.

  - **Vision-Centric Prediction with a Standard Architecture (no task-specific modules)**: Youtu-VL treats image and text tokens with equivalent autoregressive status, empowering it to perform vision-centric tasks for both dense vision prediction (e.g., segmentation, depth) and text-based prediction (e.g., grounding, detection) within a standard VLM architecture, eliminating the need for task-specific additions. This design yields a versitile general-purpose VLM, allowing a single model to flexibly accommodate a wide range of vision-centric and vsion-language requirements.

  <p align="center">
      <img src="assets/architecture.png" width="90%"/>
  <p>

<a id="benchmarks"></a>
## 🏆 Model Performance

### Vision-Centric Tasks

  <p align="center">
      <img src="assets/vision-centric-performance.png" width="90%"/>
  <p>


### General Multimodal Tasks


  <p align="center">
      <img src="assets/general-multimodal-performance.png" width="90%"/>
  <p>


<a id="quickstart"></a>
## 🚀 Quickstart

### Using Transformers to Chat

Ensure your Python environment has the `transformers` library installed and that the version meets the requirements.

```bash
pip install "transformers>=4.56.0,<=4.57.1" torch accelerate pillow torchvision git+https://github.com/lucasb-eyer/pydensecrf.git opencv-python-headless
```

The snippet below shows how to interact with the chat model using `transformers`:

```python
from transformers import AutoProcessor, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "tencent/Youtu-VL-4B-Instruct", attn_implementation="flash_attention_2", torch_dtype="auto", device_map="cuda", trust_remote_code=True
).eval()

processor = AutoProcessor.from_pretrained(
    "tencent/Youtu-VL-4B-Instruct", use_fast=True, trust_remote_code=True
)

img_path = "./assets/logo.png"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},
            {"type": "text",  "text": "Describe the image"},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

generated_ids = model.generate(
    **inputs,
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    do_sample=True,
    max_new_tokens=32768,
    img_input=img_path,
)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
outputs = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
generated_text = outputs[0]
print(f"Youtu-VL output:\n{generated_text}")
```

### Demo for VL and CV tasks

A simple demo for quick start, including VL and CV tasks.

```bash
cd demo
python demo.py
```

```bash
cd demo
jupyter notebook demo.ipynb
```

The core part of this demo is three lines below:

```python
model_path = "tencent/Youtu-VL-4B-Instruct"
youtu_vl = YoutuVL(model_path)
response = youtu_vl(prompt, img_path, seg_mode=seg_mode)
```

### Using Llama.cpp to Chat

This guide will help you quickly deploy and invoke the **Youtu-VL-4B-Instruct-GGUF** model. 

```bash
llama-server -hf tencent/Youtu-VL-4B-Instruct-GGUF:Q8_0  \
  --port 8080 \
  --image-max-tokens 2048 \
  --temp 0.1 \
  --top-p 0.001 \
  --repeat-penalty 1.05 \
  -n 12280 \
  --host 0.0.0.0
```

## TODO List

- [ ] Release evaluation codes


## 🎉 Citation

If you find our work useful in your research, please consider citing our paper:

```
@article{youtu-vl,
  title={Youtu-VL: Unleashing Visual Potential via Unified Vision-Language Supervision},
  author={Tencent Youtu Lab},
  year={2026},
  eprint={2601.19798},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2601.19798}, 
}

@article{youtu-llm,
  title={Youtu-LLM: Unlocking the Native Agentic Potential for Lightweight Large Language Models},
  author={Tencent Youtu Lab},
  year={2025},
  eprint={2512.24618},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2512.24618}, 
}
```
