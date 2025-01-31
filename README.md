# 📩 SMS Spam Detection using Machine Learning  

## 📌 Overview  
This project is a **Machine Learning-based SMS Spam Detection System** that classifies messages as **Spam or Ham (Not Spam)**. It follows a complete **end-to-end pipeline**, including **data extraction, cleaning, preprocessing, feature engineering, model ensembling, and evaluation**.  

The trained model and vectorizer are stored in separate folders as `.pkl` files and are **directly used in a Streamlit web app** for real-time SMS classification.  

---

## 📁 Project Structure  
```
📦 SMS-Spam-Detection  
 ┣ 📂 Dataset  
 ┃ ┗ 📜 spam.csv                 # SMS Spam Dataset (Kaggle)  
 ┣ 📂 Model  
 ┃ ┗ 📜 voting_classifier.pkl     # Trained Voting Classifier Model  
 ┣ 📂 Vectorizer  
 ┃ ┗ 📜 vectorizer.pkl            # Saved TF-IDF Vectorizer  
 ┣ 📜 smsspamprediction.ipynb     # Main Colab file (Data Processing + Model Training)  
 ┣ 📜 streamlitapp.py             # Streamlit Web App  
 ┣ 📜 requirements.txt            # Dependencies for the Project  
 ┗ 📜 README.md                   # Project Documentation  
```

---

## 📊 Dataset  
- The dataset (`spam.csv`) is sourced from **Kaggle** and contains labeled SMS messages as **spam or ham**.  
- It is stored in the **Dataset** folder for easy access.  

---

## 🔧 Data Preprocessing  
The following preprocessing steps were applied before training the model:  
✔️ **Text Cleaning:** Removing unwanted characters, punctuations, and special symbols.  
✔️ **Stopword Removal:** Eliminating frequently used words that do not contribute to classification.  
✔️ **TF-IDF Vectorization:** Transforming text into numerical representations.  

---

## 🤖 Machine Learning Models  
A **hard voting ensemble** was applied to combine the predictions of the following six models:  
1️⃣ **K-Nearest Neighbors (KNN)**  
2️⃣ **Support Vector Machine (SVM)**  
3️⃣ **Naïve Bayes**  
4️⃣ **Random Forest**  
5️⃣ **XGBoost**  
6️⃣ **Logistic Regression**  

---

## 📈 Model Performance  
After applying the **Voting Classifier**, the final model achieved:  
✔️ **Accuracy**: **99%** (Before ensembling: **97%**)  
✔️ **Confusion Matrix**:  
```
[[ 164    8]  
 [   2 1108]]  
```
✔️ **Precision, Recall, and F1-score**: **0.99 across all metrics**  

🚫 **Cross-validation was not used**; only a single train-test split was performed.  

---

## 🚀 How to Use  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-repo-name.git
cd SMS-Spam-Detection
```

### 2️⃣ Install Dependencies  
Before running the app, install the required libraries:  
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App  
```bash
streamlit run streamlitapp.py
```
📌 This will launch a **simple user interface** where users can enter an SMS message and check whether it is **Spam or Ham**.  

---

## 🎯 Features  
✔️ **End-to-End ML Pipeline** – Data cleaning, vectorization, model training, and evaluation in **one file** (`smsspamprediction.ipynb`).  
✔️ **Voting Classifier with 6 ML Models** for enhanced accuracy.  
✔️ **Pre-Trained `.pkl` Files** for instant usage in the Streamlit app.  
✔️ **Simple Streamlit Web App** – Accepts **single text input** and predicts Spam/Ham.  
✔️ **No Deep Learning or Pretrained Models Used** – Pure ML-based approach.  

---

## 📦 Dependencies  
The necessary Python libraries are listed in `requirements.txt`. The major dependencies include:  
- **scikit-learn** (for machine learning models)  
- **pandas, numpy** (for data handling)  
- **nltk** (for text processing)  
- **streamlit** (for web app deployment)  

Make sure to install them before running the app.  

---

## 📚 Documentation  
For more in-depth details on the project, methodology, and results, check out the following document:  
[Project Documentation (DOC)](https://docs.google.com/document/d/13N-qblxnE2BAfs45xXZQwDUMDvgI0Yl4/edit?usp=drive_link&ouid=106279639348725340140&rtpof=true&sd=true)
if any issues in Document you can refer this below pdf for better understanding
[Project Documentation (PDF)](https://drive.google.com/file/d/1Ig1EFQuKsDT8jkXAOCnxRS0k_CAa4S2X/view?usp=sharing)
---


## 📌 Notes  
- The `.pkl` files (**voting_classifier.pkl & vectorizer.pkl**) are stored in the **Model** and **Vectorizer** folders, respectively.  
- The dataset (`spam.csv`) is stored in the **Dataset** folder for easy organization.  
- The **dataset and requirements.txt** files are included for reference, but libraries must be installed **before running** `streamlitapp.py`.  

---

## 📬 Contact  
If you have any questions, feel free to reach out or raise an issue in the repository.  

---
