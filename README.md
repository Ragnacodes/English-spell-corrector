# English Spelling Correction with Deep Learning

> A full pipeline for automatic English spelling correction: Wikipedia-scale dataset generation with realistic noise injection, evaluated across four model architectures — Noisy Channel, Seq2Seq LSTM, BERT masked-language modeling, and Bidirectional LSTM.

[![Python Version](https://img.shields.io/badge/python-3.x%2B-blue.svg)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

## 📌 Overview

Spelling correction is a foundational NLP task that underpins search engines, summarization systems, and text editors. Most public benchmarks rely on small, static datasets. This project builds a large-scale, self-constructed dataset from scratch — crawling Wikipedia for 3,000+ seed words, cleaning and tokenizing the resulting text, and injecting four types of synthetic keyboard-realistic noise — then benchmarks multiple correction models against it.

The dataset contains 255,800 sentence pairs with ~5.6M words across 103,000+ unique vocabulary items. Four modeling approaches are explored and compared: a classical Noisy Channel model with edit distance, two character-level Seq2Seq LSTM architectures (TF1 and TF2), and a BERT-based masked language model correction system. A minimal FastAPI server skeleton wraps the output for deployment.

## ✨ Key Features

* **Wikipedia-scale dataset pipeline:** `crawler.py` fetches full Wikipedia articles for each word in a 3K wordlist via the Wikipedia API, yielding ~45MB of raw text. A Kaggle spell-correction word list is merged in as a secondary source.
* **Four-type noise injection:** `noise_generation.py` applies keyboard-proximity replacement (50% weight), character insertion (15%), deletion (15%), and transposition (20%) at a configurable rate (default 0.3 max per token), with length-dependent probability — shorter words are less likely to be corrupted.
* **Noisy Channel model:** Uses edit distance (dynamic programming) to find minimum-distance candidates from the NLTK English word list, then ranks them by corpus frequency via `wordfreq`. Serves as the interpretable baseline.
* **Seq2Seq LSTM spell corrector:** A character-level encoder-decoder with stacked LSTM layers, dropout, attention (via `tf.contrib.seq2seq`), and greedy decoding at inference time. Implemented in both TF1 (`SpellChecker_tf1.ipynb`) and TF2 with `tensorflow-addons` (`SpellChecker_tf2.ipynb`). The trained model is saved as `models/lstm_model.h5`.
* **Bidirectional LSTM (Noisy Channel + RNN):** `NoisyChannel_LSTM.ipynb` combines the classical Noisy Channel candidate generation with a Bidirectional LSTM sequence model, trained on 180,000 sentence pairs with character-level integer encoding.
* **BERT masked-language model correction:** `Bert.ipynb` uses Google's pretrained `uncased_L-12_H-768_A-12` BERT checkpoint. For each token, it generates edit candidates (distance ≤ 2) via a Peter Norvig-style `SpellCorrector` class, masks each candidate back into the sentence, scores them with BERT's `[MASK]` prediction softmax, and returns the highest-probability correction.
* **FastAPI serving layer:** `server.py` exposes a `PUT /correct` endpoint accepting a `Sentence` payload, ready to be backed by any of the trained models.

## 🛠️ Tech Stack

* **Core Language:** Python 3
* **Deep Learning:** TensorFlow 1.x / 2.x, Keras, tensorflow-addons
* **Pretrained Models:** BERT (`bert-tensorflow`, `uncased_L-12_H-768_A-12`), HuggingFace Transformers
* **NLP Utilities:** NLTK (`sent_tokenize`, `TreebankWordTokenizer`, stopwords), wordfreq
* **Data:** Wikipedia API (`wikipedia-api`), Kaggle Spelling Dataset
* **Serving:** FastAPI, Uvicorn
* **Visualization:** Matplotlib, NumPy

## 🚀 Getting Started

### Prerequisites

Python 3 and pip.

### Installation

```bash
git clone https://github.com/Ragnacodes/NLP-Project.git
cd NLP-Project
pip install -r requirements.txt
```

### Build the dataset from scratch

```bash
# Downloads wordlist, crawls Wikipedia, preprocesses, injects noise, and prints statistics
./run.sh
```

Or run each step individually:

```bash
python3 src/crawler.py          # Crawl Wikipedia + clean Kaggle spell list
python3 src/preprocessing.py    # Tokenize, filter, produce english_tokens.csv
python3 src/noise_generation.py # Inject noise → dataset.csv (noisy | correct pairs)
python3 src/statistics.py       # Print metrics and plot word frequency histograms
```

A sample dataset is available [on Google Drive](https://drive.google.com/drive/folders/1I4XX3PjHT88RxuCAlYhwA34n_z9djVQ9?usp=sharing).

## 💻 Usage

### Run a model notebook

Open any of the following in Jupyter or Google Colab and mount your `dataset.csv`:

| Notebook | Approach |
|---|---|
| `src/NoisyChannel_LSTM.ipynb` | Noisy Channel (edit distance + wordfreq) + BiLSTM |
| `src/SpellChecker_tf1.ipynb` | Character-level Seq2Seq LSTM (TensorFlow 1.x) |
| `src/SpellChecker_tf2.ipynb` | Character-level Seq2Seq LSTM (TensorFlow 2.x + addons) |
| `src/Bert.ipynb` | BERT masked-LM with Norvig-style candidate generation |

### Run the API server

```bash
uvicorn src.server:app --reload
# PUT /correct  →  {"sentence": "Ths is a tset sentense"}
```

## 📊 Results

### Model Comparison

| Model | Evaluation | Accuracy | Notes |
|---|---|---|---|
| **BERT (masked-LM)** | 10 held-out sentences | **60%** | Best performing; corrects multi-error sentences well |
| **Noisy Channel (edit distance + wordfreq)** | 5 held-out sentences | **0%** | Struggles with multi-word context; degrades sentences rather than correcting them |
| **Seq2Seq LSTM (TF1 / TF2)** | Qualitative (checkpoint `kp=0.75,nl=2,th=0.95`) | — | Training converges; best hyperparams: 2 layers, rnn_size=512, lr=0.0005, dropout=0.75 |
| **BiLSTM (NoisyChannel_LSTM)** | 3 epochs on 180K samples | val_loss ≈ 56.6 (MSE) | Training loss plot available in `src/NoisyChannel_LSTM.ipynb` |

### Qualitative BERT Examples

The following examples are from the `evaluate(10)` run in `src/Bert.ipynb` (format: noisy input → BERT output → ground truth):

```
"Stripping one of their own bodily agency and sexuality as ewll as humaniy"
→ "Stripping one of their own bodily agency and sexuality as well as humanity"  ✓

"The wickOt keeper wearV large webbed glvoes"
→ "The wicket keeper wears large webbed gloves"  ✓

"It can be pHepared and presented in a variety of ways"
→ "It can be prepared and presented in a variety of ways"  ✓

"nd REM sleep SleeM is divided into hwo broad typws dye movement or NREM sleep..."
→ "and REM sleep Sleep is divided into two broad types eye movement or NREM sleep..."  ✓
```

### Dataset Statistics

The word frequency histogram (with and without stop words) is plotted by `src/statistics.py`. The top repeated tokens are dominated by common function words ("the", "a", "of", "and"), which is expected for Wikipedia-sourced text. See `reports/P1_Report.md` for the full statistics output and histogram descriptions.

## 📁 Project Structure

```

├── src/
│   ├── crawler.py               # Wikipedia + Kaggle data collection
│   ├── preprocessing.py         # Sentence tokenization and token filtering
│   ├── noise_generation.py      # Keyboard-proximity noise injection (4 types)
│   ├── statistics.py            # Dataset metrics and word frequency plots
│   ├── server.py                # FastAPI /correct endpoint
│   ├── NoisyChannel_LSTM.ipynb  # Noisy Channel baseline + BiLSTM model
│   ├── SpellChecker_tf1.ipynb   # Seq2Seq LSTM (TF1)
│   ├── SpellChecker_tf2.ipynb   # Seq2Seq LSTM (TF2 + addons)
│   └── Bert.ipynb               # BERT masked-LM correction
├── models/
│   └── lstm_model.h5            # Saved Seq2Seq LSTM weights
├── data/
│   ├── wordlist.txt             # 3K seed words for crawling
│   ├── wikipedia_raw/           # Per-word Wikipedia article text files
│   ├── kaggle_spell_list/       # Raw Kaggle spelling dataset
│   └── dataset.csv              # Final (noisy_sentence, correct_sentence) pairs
├── reports/
│   └── P1_Report.md             # Phase 1 report: dataset construction and noise pipeline
├── P1_Report.pdf                # PDF version of the phase 1 report
├── requirements.txt
└── run.sh                       # End-to-end dataset build script
```
