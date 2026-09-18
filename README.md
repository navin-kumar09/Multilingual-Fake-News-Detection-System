# 🌐 Multilingual Fake News Detection & Verification

A robust, end-to-end Machine Learning and NLP pipeline designed to detect and classify fake news across multiple languages (**English, Hindi, and regional datasets**). This project combines multilingual translation support (`Helsinki-NLP`), automated language detection (`langdetect`), and classical machine learning classifiers to combat misinformation at scale.

---

## 🚀 Key Features

* **Multilingual Support & Translation:** Integrates Helsinki-NLP sequence-to-sequence models (`opus-mt-hi-en` and `opus-mt-en-hi`) to seamlessly bridge language gaps between Hindi and English text inputs.
* **Automated Language Detection:** Smart language identification with built-in heuristics to correctly handle closely related Indic language tags (e.g., distinguishing Marathi/Hindi overlaps).
* **Multi-Dataset Aggregation Pipeline:** Automatically extracts, parses, and standardizes data from several popular benchmark corpora:
  * **ISOT Fake News Dataset** (`true.csv`, `fake.csv`)
  * **IFND** (Indian Fake News Dataset)
  * **LIAR Dataset** (benchmark short statements with multi-class truth ratings)
  * **India Fake News Incidents Dataset**
  * **Hindi Real & Fake News Text Corpora**
* **Unified Data Binarization:** Standardizes multi-tiered labeling schemas into a clean binary classification format (`1` = Real, `0` = Fake).
* **Scalable ML Pipeline:** Uses TF-IDF vectorization paired with robust classifiers (Logistic Regression, Random Forest, and Gradient Boosting) for efficient text classification.

---

## 📂 Project Structure

```text
my_project/
├── dataset/                    # Extracted benchmark datasets
├── src/
│   ├── __init__.py
│   ├── config.py             # Paths, configurations, and label mappings
│   ├── data_loader.py        # Dataset ingestion, parsing, and binarization
│   ├── translation.py        # Language detection and translation wrapper
│   └── model.py              # ML pipeline, training, and evaluation
├── main.py                   # Entry point script
├── requirements.txt
└── README.md
