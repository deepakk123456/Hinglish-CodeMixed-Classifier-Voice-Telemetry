# NEURAL-CORE v3.0: Multimodal Disambiguation Pipeline for Hinglish Code-Mixed Text

An advanced dual-pathway hybrid NLP framework designed specifically for sentiment analysis and semantic mapping of noisy, Romanized Hinglish (Hindi-English code-mixed) conversational streams, featuring high-speed linear feature spaces and real-time cross-attention Transformer cores with integrated voice command telemetry.

---

## 🚀 Key Architectural Features

- **Multimodal Audio Ingestion Grid:** Captures live acoustic waves using browser/local hardware audio streams with localized `hi-IN` calibration to bypass heavy translation bottlenecks.
- **Pathway A (Sub-Word N-Gram Space):** High-speed linear feature extraction mapping character n-grams ($N=2$ to $N=6$) via TF-IDF matrices to handle extreme spelling mutations ("bohot" -> "bht", "bhut").
- **Pathway B (Transformer Core Matrix):** Dense multi-head self-attention layer utilizing multilingual foundation layers (mBERT/IndicBERT) to capture long-range contextual relationships and structural grammar shifts.
- **Ensemble Fusion Layer:** Dynamic token routing based on sequence length metrics to achieve transformer-grade accuracy under high-throughput low-latency network parameters (<15ms).

---

## 📊 System Topology & Signal Flow

### 🎙️ Audio Transcription Telemetry



1. **Acoustic Wave Capture:** Continuous hardware ingestion via PyAudio pipeline wrappers.
2. **Ambient Noise Normalization:** 0.8-second environmental calibration matrix lock.
3. **Phonetic Script Deserialization:** Regional multi-dialect pattern matching via Google Web Speech Core using localized linguistic boundaries.
4. **Buffer Synchronization:** Transcribed data payloads are safely written onto the application state vector without structural data loss.

---

## 📈 Performance & Evaluation Validation

Our hybrid implementation was extensively benchmarked against balanced code-mixed datasets consisting of social media comments and unstructured product reviews.

### 🏁 Model Benchmarks
- **Overall Pipeline Accuracy:** **91.5%**
- **Linear Path Latency:** **10.91 ms**
- **Deep Transformer Core Latency:** **91.55 ms**
- **Hybrid Optimized Target Latency:** **<15.00 ms**

---

## 🛠️ Local Machine Setup Instructions

Follow these exact configurations to deploy and initialize the application kernel locally:

### 1. Project Ingestion

git clone [https://github.com/deepakk123456/Hinglish-CodeMixed-Classifier-Voice-Telemetry.git](https://github.com/deepakk123456/Hinglish-CodeMixed-Classifier-Voice-Telemetry.git)
cd Hinglish-CodeMixed-Classifier-Voice-Telemetry

2. Dependency Resolution
Ensure you have Python 3.10+ installed. Populate your local virtual environment using the core dependencies manifest:
pip install -r requirements.txt

(Note: If PyAudio installation drops an exception on Windows environments, execute pip install pipwin followed by pipwin install pyaudio)

3. Launching the App Matrix Interface
To initialize the system web dashboard layout, run the following streaming command instance:


streamlit run app/interface.py
🔮 Future Horizon Roadmap
Integrate parameter-efficient fine-tuning (PEFT/LoRA) for dense localized weight calibration.

Scale structural pipeline boundaries horizontally to accommodate regional language variations like Beng-lish (Bengali) and Tam-lish (Tamil).

Investigator: Deepak Kainthola (MCA Student)

Institution: Amity University Punjab


### 🚀 Terminal Se GitHub Par Push Kaise Karein?

Jab aap VS Code mein ye file banakar save kar lein, toh niche VS Code ke terminal mein bas ye **3 commands** ek ke baad ek chala lijiye:

**1. Is naye README ko staging area mein add karein:**

git add README.md
2. Is change par commit seal lagayein:

git commit -m "Docs: Added hyper-detailed professional project README"
3. GitHub par push kar dein:


git push origin main
