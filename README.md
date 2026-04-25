# 🚢 Ship Trajectory Anomaly Detection (Maritime AI)

---

## 👥 Team Name

FlameTech

---

## 👨‍💻 Team Members

* Nishtha Sahani (DevOps / GitHub / Deployment)
* Vaibhav Maurya (Frontend Developer)
* Jayati Mahato (ML Engineer)
* Sahil Gazi (Backend Developer)

---

## 🎯 Problem Statement

Maritime systems generate massive amounts of ship trajectory data (AIS data), but identifying **suspicious or abnormal movement patterns** manually is extremely difficult.

This leads to challenges such as:

* Illegal fishing 🚫
* Smuggling 🚢
* Unauthorized route deviations ⚠️

---

## 💡 Solution Overview

We built an **AI-powered anomaly detection system** that:

* Analyzes ship trajectory data
* Learns normal movement patterns
* Detects unusual or suspicious behavior
* Visualizes anomalies on an interactive dashboard

👉 The system helps authorities quickly identify risky maritime activities.

---

## 🧠 System Architecture

```
ML Model → Backend API → Frontend Dashboard
```

1. ML Model analyzes trajectory data and detects anomalies
2. Backend loads the model and exposes API endpoints
3. Frontend fetches data and displays it on a map

---

## 📁 Project Structure

```
ship-anomaly-detection/

├── frontend/        # React UI (map + dashboard)
├── backend/         # FastAPI server (API layer)
├── ml-model/        # Model training & saved model
├── docs/            # Documentation
├── README.md
```

---

## 🛠️ Tech Stack

### Frontend

* React.js
* Leaflet / Mapbox (for maps)

### Backend

* FastAPI
* Python

### Machine Learning

* Scikit-learn / PyTorch
* Pandas, NumPy

### Deployment

* Vercel (Frontend)
* Render (Backend)

### Version Control

* Git + GitHub

---

## ⚡ Efficiency Focus

* Lightweight ML model for fast predictions
* API optimized for quick response
* Minimal architecture for fast deployment
* Real-time visualization (simulated if needed)

---

## 🔄 Workflow

* Each member works in their own folder
* Code is uploaded regularly to GitHub
* Integration handled by DevOps
* Backend connects ML → Frontend connects API

---

## 💻 Local Setup Instructions

### Clone Repository

```
git clone https://github.com/your-username/ship-anomaly-detection.git
cd ship-anomaly-detection
```

---

### Run Backend

```
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

---

### Run Frontend

```
cd frontend
npm install
npm start
```

---

## 🔗 API Example

### GET /anomalies

```
[
  {
    "ship_id": "A1",
    "lat": 19.07,
    "lon": 72.87,
    "anomaly_score": 0.9
  }
]
```

---

## 🚀 Future Improvements

* Real-time AIS data integration
* Advanced deep learning models (LSTM)
* Alert system (SMS / Email)
* Historical trend analysis

---

## 🏁 Conclusion

This project demonstrates how AI + full-stack development can be combined to solve **real-world maritime security problems** efficiently.

---
