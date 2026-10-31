#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
df = pd.read_csv('dataset.csv')


# In[29]:


import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
warnings.filterwarnings("ignore")


# In[30]:


df.head(10)


# In[31]:


df.shape


# In[32]:


df.tail()


# In[33]:


df.isnull().sum()


# In[34]:


df.duplicated()


# In[35]:


df.duplicated().sum()


# In[36]:


df.info()


# In[37]:


df.columns


# In[38]:


df['site_category'].unique()


# In[39]:


plt.figure(figsize=(12, 6))
sns.countplot(data=df, y='site_category')
plt.title('Distribution of addvertisement field')
plt.xlabel('Count')
plt.ylabel('Advertisement Type')
plt.tight_layout()
plt.show()


# In[40]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Compute Click-Through Rate (CTR) for each advertisement category
ctr_by_category = df.groupby("site_category")["click"].mean().sort_values(ascending=False).reset_index()

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(data=ctr_by_category, x="click", y="site_category", palette="viridis")

plt.title("Click-Through Rate (CTR) by Advertisement Category")
plt.xlabel("Average CTR (Click Rate)")
plt.ylabel("Advertisement Category")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# In[41]:


import pandas as pd
import matplotlib.pyplot as plt
# Calculate Click-Through Rate (CTR) by device type
ctr_by_device = df.groupby("device_type")["click"].mean().sort_values(ascending=False)

# Plot using Matplotlib
plt.figure(figsize=(7, 5))
plt.bar(ctr_by_device.index.astype(str), ctr_by_device.values, color='orange', edgecolor='black')

# Add titles and labels
plt.title("Click-Through Rate (CTR) by Device Type", fontsize=14, fontweight='bold')
plt.xlabel("Device Type")
plt.ylabel("Average Click Rate (CTR)")

# Add grid and value labels
plt.grid(axis='y', linestyle='--', alpha=0.6)
for i, v in enumerate(ctr_by_device.values):
    plt.text(i, v + 0.005, f"{v:.2f}", ha='center', fontsize=9)

plt.tight_layout()
plt.show()


# In[42]:


X=df.iloc[:,:-1]
y=df['site_category']


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[17]:


# Select only numerical columns for scaling
numerical_cols = X_train.select_dtypes(include=['number']).columns
X_train_numerical = X_train[numerical_cols]
X_test_numerical = X_test[numerical_cols]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_numerical)
X_test_scaled = scaler.transform(X_test_numerical)


# In[18]:


model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000)


# In[19]:


model.fit(X_train_scaled, y_train)


# In[20]:


y_pred = model.predict(X_test_scaled)


# In[21]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[22]:


cm=confusion_matrix(y_test, y_pred)


# In[23]:


sns.heatmap(cm, annot=True, fmt="d", cmap="RdBu")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix LOGISTIC REGRESSION")
plt.show()


# In[24]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Data Loading ---
df = pd.read_csv("Dataset.csv")

# --- 2. Feature Engineering & Preprocessing ---

# Drop high-cardinality/identifier columns
columns_to_drop = ['id', 'hour', 'device_id', 'device_ip']
df_processed = df.drop(columns=columns_to_drop)

# Identify the remaining categorical columns (object type)
categorical_cols = df_processed.select_dtypes(include=['object']).columns

# Apply Label Encoding to all remaining categorical features
le = LabelEncoder()
for col in categorical_cols:
    df_processed[col] = df_processed[col].astype(str)
    df_processed[col] = le.fit_transform(df_processed[col])

# --- 3. Data Splitting ---
X = df_processed.drop('click', axis=1)
y = df_processed['click']

# Split data into training and testing sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- 4. Logistic Regression Model ---

# Instantiate and train the model
log_reg = LogisticRegression(solver='liblinear', random_state=42)
log_reg.fit(X_train, y_train)

# Predict and evaluate
y_pred_lr = log_reg.predict(X_test)

print("--- Logistic Regression Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))


# In[25]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Data Loading ---
df = pd.read_csv("Dataset.csv")

# --- 2. Feature Engineering & Preprocessing ---

# Drop high-cardinality/identifier columns
columns_to_drop = ['id', 'hour', 'device_id', 'device_ip']
df_processed = df.drop(columns=columns_to_drop)

# Identify the remaining categorical columns (object type)
categorical_cols = df_processed.select_dtypes(include=['object']).columns

# Apply Label Encoding to all remaining categorical features
le = LabelEncoder()
for col in categorical_cols:
    df_processed[col] = df_processed[col].astype(str)
    df_processed[col] = le.fit_transform(df_processed[col])

# --- 3. Data Splitting ---
X = df_processed.drop('click', axis=1)
y = df_processed['click']

# Split data into training and testing sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- 4. Random Forest Model ---

# Instantiate and train the model
# n_jobs=-1 uses all available cores for faster training
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = rf_clf.predict(X_test)

print("--- Random Forest Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))


# In[26]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Using class_weight='balanced' to handle the 80/20 imbalance
rf_clf_weighted = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf_clf_weighted.fit(X_train, y_train)


# In[27]:


get_ipython().system('pip install -q streamlit pyngrok')


# In[43]:


pip install streamlit pandas numpy matplotlib seaborn scikit-learn


# In[44]:


import joblib

# 1. SAVE THE MODEL
# Assuming your final trained Random Forest model is stored in the variable 'rf_clf_weighted'
joblib.dump(rf_clf_weighted, 'final_ad_recommender_model.joblib')
print("Model saved as: final_ad_recommender_model.joblib")

# 2. SAVE THE PREPROCESSOR (REQUIRED FOR NEW DATA)
# If you used a LabelEncoder or StandardScaler on your features (X_train), you must save that object too.
# For example, if 'encoder' was your LabelEncoder object:
# joblib.dump(encoder, 'feature_encoder.joblib')


# In[ ]:




