# 📝 NLP Sentiment Analysis using Machine Learning

## 📌 Project Overview

This project is an end-to-end NLP Sentiment Analysis application that predicts whether a customer review is **Positive**, **Negative**, or **Neutral** using Machine Learning.

The project includes:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- TF-IDF Feature Extraction
- Multiple Machine Learning Models
- Model Comparison
- Streamlit Deployment
- Single Review Prediction
- Bulk File Prediction
- Automatic Rating Generation
- Recommendation System

---

## 🚀 Features

✔ Text Cleaning using NLP

✔ Lowercase Conversion

✔ URL Removal

✔ HTML Tag Removal

✔ Special Character Removal

✔ Stopword Removal

✔ Lemmatization

✔ TF-IDF Vectorization

✔ Multiple ML Models

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM
- Random Forest

✔ Model Comparison

✔ Confusion Matrix

✔ ROC Curve

✔ Sentiment Prediction

✔ Automatic Rating Generation (1–5 Stars)

✔ Recommendation Based on Sentiment

✔ CSV/Excel File Upload

✔ Download Prediction Results

---

## 📂 Project Structure

```
NLP-Sentiment-Analysis/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── reviews.csv
├── clean_reviews.csv
├── requirements.txt
├── README.md
│
├── notebooks/
│   ├── Part1_Data_Cleaning.ipynb
│   ├── Part2_Model_Building.ipynb
│   └── Part3_Deployment.ipynb
│
└── images/
```

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- WordCloud
- Streamlit
- Pickle

---

## 📊 Machine Learning Workflow

1. Load Dataset
2. Data Cleaning
3. Text Preprocessing
4. TF-IDF Vectorization
5. Train-Test Split
6. Model Training
7. Model Evaluation
8. Compare Models
9. Save Best Model
10. Deploy with Streamlit

---

## 🤖 Models Used

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM (Selected for Deployment)
- Random Forest

---

## 📈 Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Classification Report
- ROC Curve
- Model Comparison Chart

---

## ⭐ Rating System

The application automatically generates ratings based on predicted sentiment.

| Prediction | Rating |
|------------|---------|
| Positive | ⭐⭐⭐⭐⭐ (5/5) |
| Neutral | ⭐⭐⭐ (3/5) |
| Negative | ⭐ (1/5) |

---

## 💡 Recommendation System

After prediction, the application provides recommendations.

### Positive

- Thank you for your valuable feedback.
- We are glad you had a great experience.

### Neutral

- Thank you for your feedback.
- We appreciate your suggestions.

### Negative

- We are sorry for your experience.
- Please contact our support team.

---

## 📁 Supported File Formats

- CSV
- Excel (.xlsx)

---

## ▶️ Run Locally

### Clone Repository

```bash
git clone https://github.com/YourUsername/NLP-Sentiment-Analysis.git
```

---

### Move into Project

```bash
cd NLP-Sentiment-Analysis
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Download NLTK Data

```python
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

---

### Run Streamlit App

```bash
streamlit run app.py
```

---

## 📸 Application Modules

### Single Review Prediction

- Enter review text
- Predict sentiment
- Display rating
- Show recommendation

### Bulk Prediction

- Upload CSV/Excel file
- Predict sentiments
- View charts
- Download results

---

## 📌 Future Improvements

- Deep Learning (LSTM)
- BERT Transformer Model
- Multi-language Support
- Sentiment Score
- Explainable AI (SHAP)
- Live Twitter Review Analysis
- Docker Deployment
- Cloud Deployment (AWS/Azure)

---

## 👨‍💻 Author

**Suyash Pokharkar**

**Data Analyst | Data Science Enthusiast**

**Skills**

- Python
- SQL
- Power BI
- Tableau
- Machine Learning
- NLP
- Data Visualization

---

## 📜 License

This project is developed for educational and portfolio purposes.
