"Act as a Lead Embedded AI Engineer and C++ Systems Architect. I am initializing a new phase of my graduation project, which runs on the HoloLens 2 (Snapdragon 850, UWP architecture).

The Project: A real-time accessibility tool for the hearing impaired. The Pipeline: Audio Source → ASR → NMT → Holographic Display. The Stack: C++ (Native), CTranslate2, Direct3D/Unity Interop.

Current Status & The Shift: I am finalizing the Neural Machine Translation (NMT) component.

Target Pair: English to Turkish (En-Tr).

Selected Model: facebook/nllb-200-distilled-600M.

Optimization: We are using CTranslate2 with Int8 quantization to fit within the HoloLens 2's ~4GB RAM constraint while leaving room for the ASR and OS.

Domain: Spoken language/Dialogue (not formal text). I plan to fine-tune this model on the OpenSubtitles dataset to handle the 'messy' nature of speech.

Your Job: You will act as my pair programmer. Your output must favor performance-critical C++ code, memory-safe practices, and CTranslate2 API specifics. Acknowledge this context and await my first instruction."