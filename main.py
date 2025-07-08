import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
np.random.seed(42)
n_samples=500
data=pd.DataFrame({
    'Age': np.random.randint(10,80,n_samples),
    'Gender': np.random.choice([0,1], n_samples),
    'Scholarship': np.random.choice([0,1],n_samples),
    'Diabetes': np.random.choice([0,1], n_samples),
    'Alcoholism': np.random.choice([0,1],n_samples),
    'SMS':np.random.choice([0,1],n_samples),
    'WaitingDays': np.random.randint(0,100,n_samples),
    'AppType':np.random.choice([0,1],n_samples),
    'Dayofweek': np.random.choice(range(0,7), n_samples),
    'season':np.random.choice([0,1,2,3], n_samples),
    'weather': np.random.choice([0,1,2], n_samples),
})
data['NoShow']=np.where(
    (data['Age']<30)&(data['SMS']==0)&(data['WaitingDays']>20),
    1,
    np.random.choice([0,1],n_samples, p=[0.7,0.3])
)
x=data.drop('NoShow',axis=1)
y=data['NoShow']
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=42)
model=RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
y_prob=model.predict_proba(x_test)[:,1]
print("\nConfusion Matrix:\n", confusion_matrix(y_test,y_pred))
print("\nclassification report: \n", classification_report(y_test,y_pred))
print("\naccuracy:", accuracy_score(y_test,y_pred))
print("\nIntervention recommendations")
for i, prob in enumerate(y_prob):
    if prob>0.5:
        print(f"patient {i+1} - High risk of No-show (Prob: {prob:.2f}) - send reminder(SMS/call)")