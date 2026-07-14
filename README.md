# 🍫 Cocoa Butter Quality Prediction Using Machine Learning

A machine learning-based web application developed as an MCA final year project to predict the quality of cocoa butter using its physicochemical properties and processing parameters. The application provides quick and reliable quality predictions through a simple Flask-based web interface, helping support quality assessment in the food and cosmetic industries.

---

## 📖 Overview

Cocoa butter is one of the most important ingredients used in chocolate and cosmetic products. Its quality depends on several physicochemical characteristics such as fatty acid composition, melting point, moisture content, viscosity, and processing conditions.

This project applies machine learning techniques to analyze these parameters and predict the quality grade of cocoa butter. The trained model is integrated into a Flask web application where users can enter sample details and receive an instant quality prediction. Machine learning can improve consistency in quality assessment by learning patterns from historical data.

---

## ✨ Features

* Predicts cocoa butter quality using Machine Learning.
* User-friendly web interface.
* Real-time prediction results.
* Flask-based backend.
* Responsive frontend using HTML, CSS, and JavaScript.
* Easy-to-use input form for cocoa butter sample parameters.

---

## 🛠️ Technologies Used

### Frontend

* HTML5
* CSS3
* JavaScript
* Bootstrap

### Backend

* Python
* Flask

### Machine Learning

* Scikit-learn
* Pandas
* NumPy
* Pickle

---

## 📂 Project Structure

```text
cocoa-butter-prediction/
│── app.py
│── model.pkl
│── requirements.txt
│── README.md
│
├── templates/
│   ├── index.html
│   ├── predict.html
│   ├── result.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── dataset/
│   └── cocoa_butter_dataset.csv
│
└── models/
    └── trained_model.pkl
```

---

## 📊 Dataset

The model is trained using a cocoa butter dataset containing important quality-related attributes, including:

* Oleic Acid
* Stearic Acid
* Palmitic Acid
* Melting Point
* Solid Fat Content
* Moisture Content
* Viscosity
* Temperature
* Pressure
* Refining Time
* Conching Time
* Fermentation Type
* Color
* Lead (Pb)
* Cadmium (Cd)
* Triglyceride Composition (SOS, POP, POS)

These features are used to classify the overall quality of cocoa butter samples.

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/sharanya200224/cocoa-butter-prediction.git
```

### Navigate to the project folder

```bash
cd cocoa-butter-prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python app.py
```

### Open in your browser

```
http://127.0.0.1:5000/
```

---

## 🚀 How to Use

1. Launch the Flask application.
2. Open the application in your browser.
3. Enter the cocoa butter sample parameters.
4. Click the **Predict** button.
5. View the predicted cocoa butter quality.

---

## 🤖 Machine Learning Workflow

* Data Collection
* Data Preprocessing
* Feature Engineering
* Model Training
* Model Evaluation
* Prediction
* Flask Deployment

---

## 📈 Future Enhancements

* Improve prediction accuracy using advanced ML models.
* Add graphical data visualization.
* Store prediction history in a database.
* Generate downloadable quality reports.
* Deploy the application on cloud platforms.
* Integrate laboratory sensor data for automated quality assessment.

---

## 🎓 Academic Project

**Project Title:** Cocoa Butter Quality Prediction Using Machine Learning

**Course:** Master of Computer Applications (MCA)

**Project Type:** Final Year Project

---

## 👩‍💻 Author

**Sharanya B**

GitHub: https://github.com/sharanya200224

---

## 📄 License

This project is developed for academic and educational purposes.
