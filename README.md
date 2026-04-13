# Resume-Screening-System
# 🤖 AI Resume Screening System

This is an AI-based web application that helps in screening resumes automatically. It helps HR to find the best candidates based on job requirements.

---

## 🚀 Features

* Login system for user (admin / hr)
* Upload resumes using CSV or PDF
* Select job category
* Enter job description
* AI compares resumes with job description
* Shows match percentage
* Displays top candidates
* Shows matched and missing skills
* Download results in CSV and PDF

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Pandas
* PyPDF2
* Scikit-learn
* Reportlab

---

## 🧠 How it works

* First, user uploads resumes
* Then enters job description
* System cleans the text
* Uses TF-IDF to convert text into numbers
* Uses cosine similarity to compare resumes
* Gives score and ranks candidates

---

## 🔑 Login Details

Username: admin
Password: 1234

---

## ▶️ How to run

1. Install libraries

```
pip install -r requirements.txt
```

2. Run the app

```
streamlit run app.py
```

---

## 🌐 Deployment

You can deploy this project on:

* Streamlit Cloud
* Render

---

## 📌 Future Improvements

* Add signup system
* Use better AI models
* Add database
* Improve UI



## ⭐ Conclusion

This project helps to reduce manual work in hiring and makes the process faster and easier.

