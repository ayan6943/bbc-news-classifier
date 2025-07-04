# 📰 BBC News Text Classifier using Logistic Regression (TF-IDF + NLP)

A deep learning-powered multi-class **news topic classifier** trained on the BBC News dataset using **Azure ML**, **TF-IDF**, and classical NLP techniques. Deployed and tested inside Azure ML Studio with reproducible pipeline-based workflow.

> 🚀 Achieved **96.6 accuracy** across 5 news categories: business, entertainment, politics, sport, and tech.

---

## ☁️ Powered by Azure Machine Learning

- Built, trained, and evaluated entirely within **Azure ML Studio**
- Used **Compute Instance** for model training and TF-IDF preprocessing
- Leverage **Notebook Interface**, custom Python environments, and data versioning
- Explored deployment endpoint creation (but deferred to local inference to avoid cloud compute costs)

This demonstrates strong familiarity with **Azure’s cloud ML ecosystem** — critical for real-world scalable ML applications.

---

## 📂 Dataset Overview

- **Source:** [BBC News Dataset](http://mlg.ucd.ie/datasets/bbc.html)
- **Documents:** 2,225 articles from the BBC news website (2004–2005)
- **Classes:**  
  - `business`  
  - `entertainment`  
  - `politics`  
  - `sport`  
  - `tech`

> **Note:** All rights in the content of the original articles are owned by the BBC.  
> If you use this dataset, please cite:  
> D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006.

---

## 🧠 Techniques Used

- **Text Preprocessing**  
  - Lowercasing, punctuation removal  
  - Stopword removal with `nltk`  
  - Lemmatization using WordNet

- **Feature Extraction**  
  - TF-IDF Vectorization (`max_features=5000`)

- **Modeling**  
  - Logistic Regression (`max_iter=1000`)

- **Evaluation Metrics**  
  - Accuracy: **96.6%**  
  - Precision, Recall, F1-Score  
  - Confusion Matrix, Class-wise WordClouds

---

## 📊 Visualizations

- 📌 **Confusion Matrix** (Seaborn heatmap)
- 📌 **WordClouds** per category (before/after cleaning)
- 📌 Feature distribution through TF-IDF stats

> These insights help validate class consistency and interpret prediction trends.

---

## 🧪 Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/bbc-news-classifier.git
cd bbc-news-classifier

# Set up virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook
