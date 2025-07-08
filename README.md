# Appointment No-Show Predictor 

This project aims to build a predictive system that identifies patients who are likely to miss their scheduled medical appointments and recommends timely interventions to reduce no-show rates.

---

##  Problem Statement

Missed appointments in healthcare settings lead to wasted resources, increased costs, and delayed care for other patients. The goal of this project is to:

- Predict which patients are at **high risk of not attending** their appointments.
- Recommend simple interventions (such as sending SMS reminders or calls) to improve attendance rates.

---

##  Technologies Used

- **Python**
- **Pandas** & **NumPy** for data handling
- **Scikit-learn** for machine learning (Random Forest Classifier)
- **Git & GitHub** for version control

---

## Project Pipeline (Clear Architecture)

| Step | Description |
|------|-------------|
| **1. Appointment Data** | Generated **synthetic patient data** including demographics, medical history, appointment type, temporal and weather factors |
| **2. Feature Creation** | Created features like Age, Gender, SMS reminder, Waiting Days, Day of Week, Season, Weather |
| **3. Risk Prediction** | Used **Random Forest Classifier** to predict the likelihood of a no-show |
| **4. Intervention Recommendations** | Suggested actions (SMS/call) for patients with a **high risk** score based on model predictions |

---

## 🏥 Synthetic Data Features

| Feature       | Description                           |
|--------------|---------------------------------------|
| Age          | Patient's age                         |
| Gender       | 0 = Male, 1 = Female                  |
| Scholarship  | Indicates free healthcare             |
| Diabetes     | 0 or 1                                |
| Alcoholism   | 0 or 1                                |
| SMS          | Whether the patient received a reminder |
| WaitingDays  | Days between scheduling and appointment |
| AppType      | 0 = General, 1 = Specialist           |
| DayOfWeek    | Day of the appointment (0 = Monday)   |
| Season       | 0 = Winter, 1 = Spring, 2 = Summer, 3 = Fall |
| Weather      | 0 = Clear, 1 = Rainy, 2 = Stormy      |

The target variable **NoShow** indicates whether the patient missed the appointment (`1` = No-Show, `0` = Attended).

---

##  Model Performance

| Metric         | Result (Synthetic Data) |
|---------------|-------------------------|
| Accuracy      | ~66%                    |
| Recall (No-Show) | ~19–29%              |
| Precision (No-Show) | ~40–56%           |

*Note: Performance is limited due to synthetic nature of the data.*

---

##  Intervention Logic

Patients with a predicted **No-Show probability > 0.5** are flagged for intervention:
