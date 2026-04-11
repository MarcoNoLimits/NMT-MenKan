---
title: NMT MenKan
emoji: 🌐
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: En-It translation API (FastAPI + CTranslate2)
---

# NMT-MenKan

**Real-Time Neural Machine Translation for HoloLens 2** *Bridging the gap for the hearing impaired through Language.*

## 📌 Project Overview
NMT-MenKan is a high-performance, on-device NMT engine specifically optimized for the **HoloLens 2 (ARM64)**. It serves as the translation layer in a cascaded pipeline (ASR → NMT), converting English speech into Italian holographic text with minimal latency.

The project name derives from **"MenKan"** (Bambara for *hear and understand*), reflecting the use of holographic light to provide clarity to the user.

## 🛠 Technical Stack
* **Architecture:** C++ Native (UWP Compatible)
* **Model:** `facebook/nllb-200-distilled-600M`
* **Inference Engine:** [CTranslate2](https://github.com/OpenNMT/CTranslate2)
* **Quantization:** Int8 (Optimized for Snapdragon 850 / 4GB RAM)
* **Language Pair:** English (En) → Italian (It)

## 🚀 Architectural Decisions
- **Why NLLB-200?** Selected over standard Transformer models for superior handling of Italian morphology and better performance on spoken-dialogue structures.
- **Why CTranslate2?** Provides a lightweight C++ execution provider that avoids the overhead of Python/PyTorch, essential for real-time holographic rendering cycles.
- **Fine-Tuning:** The model is targeted for fine-tuning on the **OpenSubtitles** dataset to better align with "messy" real-world speech captured via ASR.

## 📂 Repository Structure
* `/src`: Native C++ implementation of the translation worker.
* `/models`: Scripts for converting HuggingFace checkpoints to CTranslate2 format.
* `/docs`: Architecture diagrams and HoloLens 2 deployment guides.

## HTTP API Deployment (Hugging Face Spaces)
- A Docker-ready HTTP API is available at `scripts/nmt_http_api.py`.
- Endpoint docs:
  - `GET /healthz`
  - `POST /translate` with JSON `{ "text": "..." }`
- Deployment guide: `HUGGINGFACE_SPACES.md`

---

## 🤖 Context for AI Assistants
*If you are assisting with this repo, please adhere to the following constraints:*

> **Role:** Lead Embedded AI Engineer & C++ Architect.
> **Target:** HoloLens 2 (Snapdragon 850, UWP).
> **Objective:** Maintain low-latency En-It translation using NLLB-200-distilled-600M via CTranslate2 (Int8). 
> **Focus:** Memory safety, SIMD optimizations for ARM64, and efficient buffer management between ASR and NMT layers.
