# Behavior-based-personalized-advertisement-recommendation
End-to-end machine learning pipeline for predicting Click-Through Rates (CTR) and delivering personalized ad recommendations using behavioral and contextual user data. Evaluates Logistic Regression, Random Forest, and XGBoost models.

Project Overview
Traditional online advertising often relies on broad demographic buckets, leading to low user engagement and ad fatigue. This project develops a supervised machine learning solution that analyzes historical browsing behavior, device characteristics, site categories, and contextual timestamps to deliver Top-N personalized ad recommendations to users.

Dataset Characteristics

Source: Synthetic dataset modeled after industrial CTR prediction benchmarks (such as the Avazu dataset structure).

Size: 1000+ ad impression records containing detailed interaction features.

Target Variable (click): Binary classification (0 = No Click, 1 = Click), featuring typical real-world class imbalance.

Key Features: site_category, device_type, device_conn_type, banner_pos, hour, and anonymized campaign attributes (C14 to C21).

Methodology and Architecture

Data Preprocessing and Cleaning: Verified data integrity, handled categorical variables via Label Encoding or One-Hot Encoding, and scaled numerical features.

Exploratory Data Analysis (EDA): Uncovered that mobile users demonstrate higher engagement rates and that central banner placements significantly impact user responsiveness.

Model Training and Evaluation: Trained and compared multiple classifiers:

Logistic Regression: For baseline interpretability and linear separability.

Random Forest and XGBoost: To capture complex, non-linear feature interactions with class weight tuning to handle data imbalance.

Recommendation Engine: Built a deployment-ready system that ranks ads and serves Top-N recommendations based on predicted click probabilities.

Tech Stack

Language: Python

Data Manipulation and Analysis: Pandas, NumPy

Machine Learning: Scikit-Learn, XGBoost

Visualization: Seaborn, Matplotlib

Environment: Jupyter Notebook or Interactive Web App

Getting Started and Installation

Clone the repository:
git clone https://github.com/Prajwal-Gurivindapalli/behavior-based-ad-recommendation.git
cd behavior-based-ad-recommendation

Install dependencies:
pip install -r requirements.txt

Run model training or launch notebooks in the notebooks directory.

Key Results and Business Impact

Models successfully outperformed random baseline guessing by capturing subtle behavioral and contextual signals.

Demonstrated that device type and ad placement are stronger predictors of engagement than demographics alone.

Provides advertisers with actionable insights to maximize campaign ROI and optimize ad spend allocation.
