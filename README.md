# Detecting Mental Health Issues Through Social Media Posts Using Machine Learning

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Data Source](#data-source)
- [Tools and Libraries](#tools-and-libraries)
- [Data Preparation and Synthetic Data Generation](data-preparation-and-synthetic-data-generation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Development](#model-development)
- [Model Evaluation](#model-evaluation)
- [Model Performance Summary](#model-performance-summary)
- [Deployment](#deployment)
- [Ethics and Data Management](#ethics-and-data-management)
- [Practical Applications](#practical-applications)
- [Conclusion](#conclusion)
- [Recommendations](#recommendations)
- [References](#references)

## Project Overview
This project aims to detect indicators of mental health issues through social media posts using machine learning. The study compares Support Vector Machine (SVM) and Bidirectional Long Short-Term Memory (BiLSTM) models to classify text as indicative or non-indicative of mental health concerns.
The project emphasizes ethical AI, data management, and real-time analysis using a Python Shiny web application.

## Problem Statement
Mental health challenges are often expressed subtly in online conversations. Early detection through data analytics can support intervention and awareness. However, real-world data poses ethical, privacy, and imbalance challenges.
This project focuses on designing an ethically compliant, high-performing ML pipeline to detect mental health signals in text while ensuring data privacy and transparency.

## Data Source
Secondary dataset: Publicly available text-based social media dataset related to mental health [Download here](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health?resource=download)

Synthetic dataset: Generated from the secondary dataset using controlled augmentation to preserve structure while ensuring privacy.

## Tools and Libraries
### Programming Language: Python
### Key Libraries:
- pandas, numpy — Data cleaning and manipulation
- scikit-learn — Machine learning (SVM, evaluation)
- tensorflow, keras — Deep learning (BiLSTM)
- nltk, re, wordcloud — Text processing and visualization
- matplotlib, seaborn — EDA and plotting
- shiny (Python) — App deployment

## Data Preparation and Synthetic Data Generation
- Cleaned text data by removing URLs, emojis, and stopwords.
- Tokenized and lemmatized posts for NLP readiness.
- Generated synthetic dataset from the secondary dataset using text augmentation (synonym replacement and paraphrasing).
- Split data into training (80%) and testing (20%) sets.
- Ensured all data met ethical and privacy compliance.

## Exploratory Data Analysis
- Examined text length, word frequency, and sentiment distribution.
- Visualized common words in positive and negative classes.
- Identified slight class imbalance, which was mitigated using oversampling.
<img width="812" height="615" alt="Screenshot 2025-03-26 151937" src="https://github.com/user-attachments/assets/46af42e2-4953-4391-b33c-a7e1757361c0" />

## Model Development
Built and compared the following models:
- Support Vector Machine (SVM): Used TF-IDF features for text vectorization.
- Bidirectional LSTM (BiLSTM): Used word embeddings with LSTM layers to capture semantic context.

## Model Evaluation
Evaluation metrics:
- Accuracy
- F1-score

## Model Performance Summary
|Model |Accuracy|F1-score|
|----- |--------|--------|
|SVM   |74.7%   |74.1%   |
|BiLSTM|71.5%   |70.8%   |

SVM outperformed BiLSTM, showing better generalization and faster training time.

## Deployment
The best-performing model (SVM) was integrated into a Python Shiny web app, enabling real-time sentiment and trend analysis of text data.
Users can input a post and receive immediate classification results with confidence scores.

#### Before a post was analyse
<img width="1919" height="818" alt="Screenshot 2025-04-24 193129" src="https://github.com/user-attachments/assets/b88aefe1-14e8-43f9-a7d6-99dc3f6b61ed" />

#### After a post was analysed
<img width="1919" height="811" alt="Screenshot 2025-04-24 193401" src="https://github.com/user-attachments/assets/facae6c3-9f66-4ff1-b777-50c7238845d2" />

## Ethics and Data Management
- Used synthetic data to preserve privacy and comply with research ethics.
- Implemented version control for datasets and models.
- Documented all preprocessing steps for reproducibility.
- Ensured transparency in model training and evaluation.

## Practical Applications
- Mental health monitoring: Early detection for online support systems.
- Social media analytics: Identifying well-being trends.
- Research support: Framework for ethical AI in health-related NLP studies.

## Conclusion
The project demonstrated how machine learning and deep learning can be ethically applied to detect mental health signals.
The SVM model proved most effective, achieving strong accuracy and interpretability.
The deployment through a Python Shiny web app provides an accessible tool for real-time analysis.

## Recommendations
- Integrate more advanced NLP models (e.g., BERT, RoBERTa).
- Increase data diversity using multilingual text.
- Expand deployment to include automated report generation and alert systems.

## References
1. [Adlung, L., Cohen, Y., Mor, U. and Elinav, E](https://doi.org/10.1016/j.medj.2021.04.006)
2. [VanderPlas, J.](https://jakevdp.github.io/PythonDataScienceHandbook/)
3. [Ho, C.T.](https://www.youtube.com/watch?v=f_Jzg8InF0g )




