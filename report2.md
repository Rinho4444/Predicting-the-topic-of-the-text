# Predicting the Topic of the Text Report
Author: HVAInternational (Pham Dang Hung, Pham Ha Khanh Chi, Le Xuan Trong, and Le Ky Nam)

## 1. Abstract
This project aims to accurately predict the topic of text using NLP. We utilize AutoGluon, with vectorization by BoW Bigram (`max_features = 18500`), for model optimization. Our approach improves on baseline models, achieving a higher Macro F1 score and better generalization.

## 2. Introduction
This project was initially developed as an assignment for the AI4B class, where we were tasked with building a predictive model for text classification. While fulfilling the course requirements, our team recognized the broader significance of this problem in the real industry.

To create the best project, the team members optimized teamwork and communication to support each other. The tasks were divided as follows:
- **Validate and preprocess data:** Pham Ha Khanh Chi
- **Building and optimizing the models:** Le Ky Nam
- **Writing the report:** Le Xuan Trong
- **Finding reference documents:** Pham Dang Hung

Each task was completed successfully with contributions from all team members. Our primary objective for this project was to successfully build a model using the information provided while applying all the knowledge we acquired in the AI4B class. Additionally, we improved our teamwork and communication skills to collaboratively solve bugs, obstacles, or issues encountered during the process.

## 3. Data Analysis
### 3.1. Dataset Overview
The dataset used was **20newsgroups**, provided by our instructors in the AI4B course.

### 3.2 Exploratory Data Analysis (EDA)
To better understand our dataset and prepare it for modeling, we conducted several visual analyses:

#### Distribution of Categories Before Preprocessing
![distribution_of_categories_before_preprocessing](https://github.com/user-attachments/assets/9f2340d1-cca9-4cde-bd94-43a42d6784b8)

##### Comments:
- The class distribution is relatively balanced (750-820 samples per class).
- However, some classes, like “talk.religion.misc,” only contain about 500 samples. This could introduce bias into the model.
- While the difference is not extreme, for models like SVM or Logistic Regression, class weighting is required.

#### Token Length Before Preprocessing
![token_length_before_preprocessing](https://github.com/user-attachments/assets/074274ca-69bb-419e-9d7a-2cb33f5ffc48)

##### Comments:
- Most token lengths are concentrated around 0-1000.
- Some extreme outliers reach up to 80,000 tokens.
- To address this, we preprocess the data by removing unnecessary tags, stopwords, and outliers.

#### Word Cloud Before Preprocessing
![word_clouds](https://github.com/user-attachments/assets/060ce787-5244-413f-888c-8af925f76f67)

##### Comments:
- Word clouds help identify category-specific words.
- For example, “rec.sport.hockey” has frequent terms like “game” and “team.”
- Noisy words such as “subject” appear frequently across categories, so we add them to `custom_stopwords`.

#### Top 20 Most Common Words Before Preprocessing
![top_20_common_words_before_preprocessing](https://github.com/user-attachments/assets/4760f74f-1d18-4003-98d3-3cc7e3230ad0)

##### Comments:
- Generic words like “like,” “one,” “don’t,” and “get” dominate.
- These words are added to the `custom_stopwords` list to reduce noise.

#### Top 20 Most Common Bigrams Before Preprocessing
![top_20_most_common_bigrams](https://github.com/user-attachments/assets/f0800fab-0f4e-4a14-b3df-1dbfb6f38608)

##### Comments:
- Some uninformative bigrams (e.g., “ax ax,” “ac uk,” “cmu edu”) are removed.
- Useful bigrams (e.g., “mit edu” for science-related topics, “nasa gov” for space-related topics) are retained.

#### Histogram of Token Lengths After Preprocessing
![token_length_after_preprocessing](https://github.com/user-attachments/assets/b3a7bffb-e244-44e4-ac9b-7bc58152f53e)

##### Comments:
- Token lengths remain concentrated around 0-1000.
- Extreme outliers have been reduced, with the longest now around 6000.

### 3.3. Data Preprocessing
1. **Standardization:** Converting text to lowercase, removing special characters and punctuation, tokenization, and stopword removal (`stopwords` from nltk and `custom_stopwords`).
2. **Vectorization:** Experimenting with different methods including Bag of Words (BoW), One-Hot Encoding, TF-IDF, and Word Embeddings (Word2Vec, FastText, etc.).

## 4. Modeling
### 4.1. Baseline Models
The following models were tested:
- Fully Connected Neural Network (FCNN)
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)
- Transformer-based models

### 4.2. Optimizing the Best Baseline Model
CNN produced the highest accuracy among the baseline models. Based on insights from [this post](https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist), we optimized CNN further.

### 4.3. Advanced Model
We used [AutoGluon for text](https://auto.gluon.ai/stable/tutorials/multimodal/text_prediction/beginner_text.html) to automate model selection and hyperparameter tuning. AutoGluon significantly outperformed all baseline models.

## 5. Model Results

| Model | Macro F1 Score (Validation) | Macro F1 Score (Test) | Training Time |
|---------|------|------|------|
| XGBoost + BoW Bigram 7500 | 0.85 | 0.7532 | - |
| CNN | 0.8729 (avg over 5 runs) | 0.7529 (avg over 5 runs) | - |
| FCNN (1 dense layer) | 0.8654 (avg over 5 runs) | 0.7621 (avg over 5 runs) | - |
| AutoGluon + BoW Bigram 18500 | **0.9145** | **0.8425** | - |

## 6. Conclusion
AutoGluon, with BoW Bigram (`max_features = 18500`), successfully optimized performance, achieving higher accuracy than traditional models. Future directions include:
- Expanding the dataset for improved generalization.
- Deploying the model using FastAPI or Streamlit to create a real-world application.
- Exploring other advanced NLP techniques like self-supervised learning.

This project has reinforced our understanding of NLP, deep learning, and teamwork. We look forward to further improving and deploying this model in real-world scenarios.

