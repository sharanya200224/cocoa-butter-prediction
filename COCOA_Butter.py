#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


df=pd.read_csv(r"C:\Users\anany\Downloads\cocoa_butter_dataset_500_unique_rows.csv")


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.describe()


# In[9]:


df.sample()


# In[10]:


df.info()


# In[11]:


df.shape


# In[12]:


df.dtypes


# In[13]:


df.isna().sum()


# In[14]:


df.columns


# In[15]:


df.rename(columns={
    'Sample ID': 'Sample_ID',
    'Oleic Acid (%)': 'Oleic_Acid',
    'Stearic Acid (%)': 'Stearic_Acid',
    'Palmitic Acid (%)': 'Palmitic_Acid',
    'Melting Point (°C)': 'Melting_Point',
    'Solid Fat Content (%)': 'Solid_Fat_Content',
    'Moisture Content (%)': 'Moisture_Content',
    'Pb (ppm)': 'Lead_ppm',
    'Cd (ppm)': 'Cadmium_ppm',
    'Viscosity (mPa·s)': 'Viscosity',
    'NIR Absorbance': 'NIR_Absorbance',
    'FTIR Transmittance': 'FTIR_Transmittance',
    'Temp (°C)': 'Temperature',
    'Pressure (bar)': 'Pressure',
    'Refining Time (hrs)': 'Refining_Time',
    'Conching Time (hrs)': 'Conching_Time',
    'Fermentation Type': 'Fermentation_Type',
    'Quality Grade': 'Quality_Grade',
    'colors': 'Color'
}, inplace=True)


# In[16]:


df.columns


# In[17]:


df.drop(columns=['Sample_ID'], inplace=True)


# In[18]:


df.head()


# In[19]:


import pandas as pd

# Load your dataset (replace 'your_file.csv' with your actual file name)


# Get minimum values of each column
min_values = df.min(numeric_only=True)

# Get maximum values of each column
max_values = df.max(numeric_only=True)

# Combine min and max into one DataFrame
min_max_df = pd.DataFrame({'Min': min_values, 'Max': max_values})

# Display the result
print(min_max_df)


# In[20]:


df['Fermentation_Type'].unique()
df['Fermentation_Type']=df['Fermentation_Type'].map({'Natural':0,'Controlled':1})
df['Fermentation_Type']


# In[21]:


df['Quality_Grade'].unique()
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df['Quality_Grade']=lb.fit_transform(df['Quality_Grade'])
df['Quality_Grade'].unique()


# In[22]:


df['Color'].unique()
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df['Color']=lb.fit_transform(df['Color'])
df['Color'].unique()


# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(df['Oleic_Acid'], bins=20, kde=True, color='chocolate')
plt.title('Distribution of Oleic Acid Content')
plt.xlabel('Oleic Acid (%)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()


# In[24]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Oleic_Acid', y='Stearic_Acid', 
                hue='Quality_Grade', palette=['red', 'orange', 'green'],
                s=100, alpha=0.7)
plt.title('Oleic Acid vs Stearic Acid by Quality Grade')
plt.xlabel('Oleic Acid (%)')
plt.ylabel('Stearic Acid (%)')
plt.legend(title='Quality Grade')
plt.grid(True, alpha=0.3)
plt.show()


# In[25]:


plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Quality_Grade', y='Melting_Point', 
            palette=['#FF9999', '#FFE888', '#88FF99'])
plt.title('Melting Point Distribution by Quality Grade')
plt.xlabel('Quality Grade')
plt.ylabel('Melting Point (°C)')
plt.grid(True, axis='y', alpha=0.3)
plt.show()


# In[26]:


plt.figure(figsize=(12, 6))
plt.plot(df['Refining_Time'], df['Viscosity'], 'o-', color='brown', 
         label='Viscosity')
plt.plot(df['Refining_Time'], df['NIR_Absorbance'], 's--', color='green',
         label='NIR Absorbance')
plt.title('Process Parameters vs Refining Time')
plt.xlabel('Refining Time (min)')
plt.ylabel('Parameter Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# In[27]:


quality_counts = df['Quality_Grade'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(quality_counts, labels=quality_counts.index, 
        autopct='%1.1f%%', startangle=90,
        colors=['#FF6B6B', '#FFD166', '#06D6A0'],
        explode=(0.05, 0.05, 0.05))
plt.title('Quality Grade Distribution')
plt.show()


# In[28]:


contaminants = df[['Lead_ppm', 'Cadmium_ppm']].mean()
plt.figure(figsize=(8, 6))
bars = plt.bar(contaminants.index, contaminants.values, 
               color=['#FF9AA2', '#FFB7B2'])
plt.title('Average Heavy Metal Content')
plt.ylabel('ppm (parts per million)')
plt.ylim(0, 0.1)
plt.grid(True, axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom')

plt.show()


# In[29]:


# Final feature selection
essential_features = [
    
    'Oleic_Acid', 'Stearic_Acid', 'Palmitic_Acid', 'SOS', 'POP', 'POS','Melting_Point', 'Solid_Fat_Content', 'Viscosity','Lead_ppm', 'Cadmium_ppm',
    'Refining_Time', 'Conching_Time','NIR_Absorbance', 'FTIR_Transmittance','Fermentation_Type'  
]

# Target variable
target = 'Quality_Grade'  # (Low/Medium/High)

# Create final dataset
X = df[essential_features]
y = df[target]

# Verify feature correlations
plt.figure(figsize=(12,10))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()


# In[30]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[31]:


X_train.shape


# In[32]:


X_test.shape


# In[33]:


y_train.shape


# In[34]:


y_test.shape


# In[35]:


import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

# 1. Initialize XGBoost without early stopping for CV
cv_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    eval_metric='mlogloss',
    random_state=42,
    max_depth=5,
    learning_rate=0.1,
    n_estimators=200
)

# 2. Cross-validation (without early stopping)
cv = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring='accuracy')
print(f"Cross-validated Accuracy: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")

# 3. Now train final model with early stopping
final_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    eval_metric='mlogloss',
    early_stopping_rounds=10,
    random_state=42,
    max_depth=5,
    learning_rate=0.1,
    n_estimators=200
)

# Convert data to DMatrix (optimized for XGBoost)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Train with validation set
final_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)

# 4. Evaluate
y_pred = final_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

# Confusion Matrix
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
plt.title('Quality Grade Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Feature Importance
plt.figure(figsize=(10,6))
xgb.plot_importance(final_model, max_num_features=15, height=0.8)
plt.title('Feature Importance')
plt.show()

# Save model
import joblib
joblib.dump(final_model, 'cocoa_quality_xgb_final.pkl')


# In[36]:


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Create preprocessing pipeline
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # SVM requires feature scaling
    ('svm', SVC(
        kernel='rbf',              # Radial Basis Function kernel
        C=1.0,                    # Regularization parameter
        gamma='scale',            # Kernel coefficient
        decision_function_shape='ovr',  # One-vs-Rest for multiclass
        probability=True,          # Enable probability estimates
        random_state=42
    ))
])

# 2. Train the model
svm_pipeline.fit(X_train, y_train)

# 3. Evaluate
y_pred = svm_pipeline.predict(X_test)
y_proba = svm_pipeline.predict_proba(X_test)  # Class probabilities

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

# Confusion Matrix
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
plt.title('SVM Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Save model
import joblib
joblib.dump(svm_pipeline, 'cocoa_quality_svm.pkl')

# Cross-validation (optional)
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(svm_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validated Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")


# In[37]:


from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Initialize and train Decision Tree
dt_model = DecisionTreeClassifier(
    criterion='gini',          # or 'entropy'
    max_depth=5,               # Prevent overfitting
    min_samples_split=10,      # Minimum samples to split
    min_samples_leaf=5,        # Minimum samples per leaf
    class_weight='balanced',   # Handle class imbalance
    random_state=42
)

dt_model.fit(X_train, y_train)

# 2. Evaluate
y_pred = dt_model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

# 3. Visualize the tree
plt.figure(figsize=(20,12))
plot_tree(dt_model, 
          feature_names=X_train.columns,
          class_names=['Low', 'Medium', 'High'],
          filled=True,
          rounded=True,
          proportion=True,
          fontsize=10)
plt.title("Cocoa Butter Quality Decision Tree")
plt.show()

# 4. Text representation (for rule extraction)
tree_rules = export_text(dt_model, 
                        feature_names=list(X_train.columns))
print("Decision Rules:\n", tree_rules)

# 5. Feature Importance
plt.figure(figsize=(10,6))
importances = dt_model.feature_importances_
sorted_idx = importances.argsort()[::-1]
plt.barh(X_train.columns[sorted_idx][:10], importances[sorted_idx][:10])
plt.xlabel("Gini Importance")
plt.title("Top 10 Important Features")
plt.gca().invert_yaxis()
plt.show()

# 6. Confusion Matrix
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 7. Save model
import joblib
joblib.dump(dt_model, 'cocoa_quality_decision_tree.pkl')


# In[38]:


import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace with your actual metrics)
metrics = {
    'Accuracy': {'Decision Tree': 0.90, 'SVM': 0.88, 'XGBoost': 0.92},
    'F1 Score': {'Decision Tree': 0.89, 'SVM': 0.88, 'XGBoost': 0.91},
    'Precision': {'Decision Tree': 0.89, 'SVM': 0.88, 'XGBoost': 0.91},
    'Recall': {'Decision Tree': 0.89, 'SVM': 0.88, 'XGBoost': 0.91}
}

# Plot setup
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.2
index = np.arange(len(metrics))
colors = ['#4CAF50', '#2196F3', '#FFC107']  # Green, Blue, Amber

# Create bars for each model
for i, (model, color) in enumerate(zip(['Decision Tree', 'SVM', 'XGBoost'], colors)):
    scores = [metrics[m][model] for m in metrics]
    ax.bar(index + i*bar_width, scores, bar_width, 
           label=model, color=color, edgecolor='grey')

# Formatting
ax.set_xlabel('Metrics', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14, pad=20)
ax.set_xticks(index + bar_width)
ax.set_xticklabels(metrics.keys())
ax.set_ylim(0.8, 1.0)
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

# Add value labels
for i, model in enumerate(['Decision Tree', 'SVM', 'XGBoost']):
    for j, metric in enumerate(metrics.keys()):
        height = metrics[metric][model]
        ax.text(j + i*bar_width, height + 0.01, f'{height:.2f}', 
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.grid(axis='y', alpha=0.3)
plt.show()


# In[39]:


class LabRefiningOptimizer:
    """Simulates how ML recommendations translate to lab actions"""
    
    def __init__(self):
        self.lab_equipment = {
            'roller_mill': {'max_temp': 90, 'time_step': 5},
            'conching_machine': {'pressure_range': (0.5, 2.5)}
        }
    
    def validate_ml_recommendation(self, ml_suggestion):
        """Checks if ML suggestions are physically feasible"""
        if ml_suggestion['temp'] > self.lab_equipment['roller_mill']['max_temp']:
            return {"error": f"Temperature exceeds equipment limit ({self.lab_equipment['roller_mill']['max_temp']}°C)"}
        
        return {
            'approved': True,
            'lab_instructions': [
                f"ADJUST roller mill to {ml_suggestion['temp']}°C",
                f"SET timer to {ml_suggestion['time']} minutes",
                "MONITOR viscosity every 5 minutes"
            ]
        }

# Example Usage
ml_output = {'temp': 85, 'time': 47}  # From your model
optimizer = LabRefiningOptimizer()
print(optimizer.validate_ml_recommendation(ml_output))


# In[40]:


import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Assume your DataFrame is already loaded and preprocessed as 'df'

# Label encode the categorical features (if not already done)
label_encoder = LabelEncoder()
if df['Fermentation_Type'].dtype == 'object':
    df['Fermentation_Type'] = label_encoder.fit_transform(df['Fermentation_Type'])

# Encode target if needed
if df['Quality_Grade'].dtype == 'object':
    df['Quality_Grade'] = label_encoder.fit_transform(df['Quality_Grade'])

# Feature matrix and target variable
X = df.drop(columns=['Quality_Grade'])
y = df['Quality_Grade']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier(
    objective='multi:softmax',       # for multi-class classification
    num_class=len(set(y)),           # number of unique classes
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Feature Importance Plot ---
plt.figure(figsize=(12, 6))
xgb.plot_importance(model, importance_type='gain', max_num_features=15)
plt.title('Top Feature Importances - XGBoost (by Gain)')
plt.tight_layout()
plt.show()

# --- Save importance values as CSV (optional) ---
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Save to CSV
importance_df.to_csv("feature_importance.csv", index=False)

# Display top features in console
print("\nTop Features Based on Importance:\n")
print(importance_df.head(10))


# In[41]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'df' is your preprocessed DataFrame

# Convert encoded target (if needed) to readable class names
# Optional: If you used LabelEncoder and want readable labels
# label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
# df['Quality_Grade'] = df['Quality_Grade'].map(label_map)

# --- 1. Boxplot for key features vs Quality Grade ---
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Quality_Grade', y='Oleic_Acid')
plt.title('Oleic Acid % vs Quality Grade')
plt.xlabel('Quality Grade')
plt.ylabel('Oleic Acid (%)')
plt.tight_layout()
plt.show()

# --- 2. Scatter Plot: Melting Point vs Stearic Acid colored by Quality Grade ---
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Melting_Point', y='Stearic_Acid', hue='Quality_Grade', palette='Set2')
plt.title('Melting Point vs Stearic Acid by Quality')
plt.tight_layout()
plt.show()

# --- 3. Violin Plot for Fat Content by Quality Grade ---
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Quality_Grade', y='Solid_Fat_Content', inner='quartile')
plt.title('Solid Fat Content Distribution by Quality Grade')
plt.tight_layout()
plt.show()

# --- 4. Heatmap of Feature Correlations ---
plt.figure(figsize=(14, 10))
corr = df.drop(columns=['Quality_Grade']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()


# In[42]:


import pandas as pd
import joblib

# Load your trained model
model = joblib.load("cocoa_quality_xgb_final.pkl")

# Define input sample with EXACT same 16 columns
sample_data = {
    'Oleic_Acid': 33.5,
    'Stearic_Acid': 27.8,
    'Palmitic_Acid': 24.2,
    'SOS': 22,
    'POP': 18,
    'POS': 29,
    'Melting_Point': 33.5,
    'Solid_Fat_Content': 55.0,
    'Viscosity': 2.7,
    'Lead_ppm': 0.04,
    'Cadmium_ppm': 0.02,
    'Refining_Time': 5,
    'Conching_Time': 6,
    'NIR_Absorbance': 1.6,
    'FTIR_Transmittance': 0.91,
    'Fermentation_Type': 1  # already encoded
}

sample_df = pd.DataFrame([sample_data])

# Predict
prediction = model.predict(sample_df)[0]
labels = ['Low', 'Medium', 'High']
predicted_label = labels[int(prediction)]

print("Predicted Quality Grade:", predicted_label)

# --- Refinement Recommendation ---
def get_refinement_recommendation(row, label):
    thresholds = {
        'Oleic_Acid': 35,
        'Moisture_Content': 0.2,  # not in model, can remove if needed
        'Melting_Point': 34,
        'Viscosity': 2.5,
        'Lead_ppm': 0.05,
        'Cadmium_ppm': 0.03
    }

    if label == "High":
        return "No refining needed. Suitable for premium cosmetic use."

    suggestions = []
    if row['Oleic_Acid'] < thresholds['Oleic_Acid']:
        suggestions.append("increase oleic acid")
    if row['Melting_Point'] < thresholds['Melting_Point']:
        suggestions.append("increase melting point")
    if row['Viscosity'] > thresholds['Viscosity']:
        suggestions.append("improve viscosity")
    if row['Lead_ppm'] > thresholds['Lead_ppm'] or row['Cadmium_ppm'] > thresholds['Cadmium_ppm']:
        suggestions.append("remove heavy metal impurities")

    return f"Refining suggested to: {', '.join(suggestions)}" if suggestions else "Refining recommended for minor improvement."

# Show recommendation
recommendation = get_refinement_recommendation(sample_df.iloc[0], predicted_label)
print("Recommendation:", recommendation)


# In[43]:


import pickle

with open("xgboost_model.pkl", "wb") as f:
    pickle.dump(model, f)


# In[44]:


import pickle

try:
    with open("xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


# In[45]:


import pickle

# Save your XGBoost model
with open('xgboost_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved successfully as 'xgboost_model.pkl'")


# In[90]:


import os
print("Current Jupyter Notebook folder:", os.getcwd())


# In[92]:


import pickle
import os

# First, check if your model variable exists
print("Current folder:", os.getcwd())

# Save your model (use YOUR model variable name)
# If your model is called 'model', 'xgb_model', 'classifier', etc. - replace below
with open('xgboost_model.pkl', 'wb') as file:
    pickle.dump(model, file)  # Change 'model' to your actual variable name

print("✅ Model saved to:", os.path.abspath('xgboost_model.pkl'))


# In[94]:


import shutil
import os

# Source (where model is now)
source = r'C:\Users\anany\xgboost_model.pkl'

# Destination (Flask app folder)
destination = r'C:\Users\anany\OneDrive\Documents\cocoa_butter_ananlysis\xgboost_model.pkl'

# Copy the file
shutil.copy2(source, destination)

# Verify it worked
if os.path.exists(destination):
    print("✅ Model successfully copied to Flask folder!")
    print(f"📍 Location: {destination}")
    print(f"📦 File size: {os.path.getsize(destination)} bytes")
else:
    print("❌ Copy failed. Please check the path.")


# In[96]:


import os

flask_folder = r'C:\Users\anany\OneDrive\Documents\cocoa_butter_ananlysis'

print("Files in Flask folder:")
for file in os.listdir(flask_folder):
    if file.endswith('.pkl'):
        print(f"  📁 {file}")


# In[ ]:




