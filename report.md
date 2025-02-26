# Predicting the topic of the text report
Author: HVAInternational (Pham Dang Hung, Pham Ha Khanh Chi, Le Xuan Trong, and Le Ky Nam)

## 1. Abstract
This project aims to accurately predict the topic of the text using NLP. We utilize _____ for model optimization. Our approach improves on baseline models, achieving a higher Marco F1 score and better generalization. 

## 2. Introduction
This project was initially developed as an assignment for the AI4B class, where we were tasked with building a predictive model for the topic of the text. While fulfilling the course requirements, our team recognized the broader significance of this problem in the real industry. _____________.

To create the best project, the team members were encouraged to optimise their teamwork and communication to help each other. The tasks were divided as follow:
- Validate and preprocess data: Pham Ha Khanh Chi
- Building and Optimising the models: Le Ky Nam
- Writing the report: Le Xuan Trong
- Finding documents: Pham Dang Hung
Each task was completed successfully by not just the assigned member but also by the help of the whole team during the process. The AIM for this project was for us to successfully build a model by the information given, applying all the knowledge we had been taught in AI4B class. Plus, we also learned teamwork and communication skills to be able to solve any bugs, obstacles or, problems passing by during the process.


## 3. Data Analysis
### 3.1. Dataset Overview
The data set that we used was **20newsgroups** provided by our teachers at the AI4B course.

### 3.2 Exploratory Data Analysis (EDA) 
To understand our dataset better and prepare it for modeling, we conducted several visual analyses:

#### Distribution of categories before preprocessing before preprocessing
![distribution_of_categories_before_preprocessing](https://github.com/user-attachments/assets/9f2340d1-cca9-4cde-bd94-43a42d6784b8)

##### Comments:
- The class distribution is relatively balance (750-820).
- However, there’s classes like “talk.religion.misc” that only have about 500. This could cause the model to have bias.
- The difference, though, is not that high. For tree-based model, this should be a problem. But, if using SVM or Logistic Regression, then we had to apply class weight.

#### Token length before preprocessing before preprocessing
![token_length_before_preprocessing](https://github.com/user-attachments/assets/074274ca-69bb-419e-9d7a-2cb33f5ffc48)

##### Comments:
- Most of the token length concentrated around 0-1000
- Most text are relatively short though there are extreme outliers that led up to 80000
- To combat this we may preprocess the data(removing tags, stopwords,...) or directly remove outliers.

#### Word cloud before preprocessing
![word_clouds](https://github.com/user-attachments/assets/060ce787-5244-413f-888c-8af925f76f67)

##### Comments:
- We create word cloud image for each class(these are some example) to figure out some category-specific words/
- Like in “rec.sport.hockey” has game, team that has a large frequency
- We will also use this to find relative words that may cause noise so we could add to custom_stopwords, like “subject” appears in 4 pictures here with a relative frequency.

#### Top 20 most common words before preprocessing
![top_20_common_words_before_preprocessing](https://github.com/user-attachments/assets/4760f74f-1d18-4003-98d3-3cc7e3230ad0)

##### Comments:
- We print out common words then
- Clearly, the common words now are quite generic: like, one, dont, get, x, also, would,...
- We would add this to the `custom_stopword` to prevent noise.

#### Top 20 most common bigrams before preprocessing
![top_20_most_common_bigrams](https://github.com/user-attachments/assets/f0800fab-0f4e-4a14-b3df-1dbfb6f38608)

##### Comments:
- There are some unmeaningful bigram like: ax ax, ac uk, don know, cmu edu, netcom com,... That needs to be remove
- But there are also some useful: mit edu(science relate), nasa gov(sci.space relate), ax max(appear a lot in comp.os.ms-window.misc)

#### Histogram of token lengths after preprocessing
![token_length_after_preprocessing](https://github.com/user-attachments/assets/b3a7bffb-e244-44e4-ac9b-7bc58152f53e)

##### Comments:
- The token length are still highly concentrated around 0-1000.
- But the extreme has relatively been remove with the longest to around 6000 to not risk losing importance context we will keep these.


#### Distribution of categories after preprocessing
![token_length_after_preprocessing](https://github.com/user-attachments/assets/b5e52797-8876-4e0c-8b64-25fc0f2a9c42)

##### Comments:
- We were worry that the aggressive preprocessing would affect the distribution but the trend seems to maintain.
- The lowest with around 500 text and the highest about 820.


#### Top 25 correlated words after preprocessing
![top_25_correlated words](https://github.com/user-attachments/assets/97902b4a-fee7-4926-af4a-756ed611a47c)

##### Comments:
- This is the top word with high correlation to a category.
- Words with high correlation to specific labels indicate key terms that define a class.
- Example: If the word "distribution" appears strongly correlated with a particular label, that class might relate to software distribution or a relevant topic.
- There are words that show correlation to all but not one category, this may affect the model as it is not specific and may cause confusion.

### 3.3. Data Preprocessing

1. Chuẩn hóa dữ liệu: chuyển về chữ thường (lowercase), loại bỏ các ký tự đặc biệt, dấu chấm câu (punctuation), tokenization, và loại bỏ stopwords bằng `stopwords` của nltk và ```custom_stopword = ["like", "one", "dont", "get", "x", "also", "would"]```.
2. Sau đó, thử nghiệm vector hóa dữ liệu bằng Bag of Words, One hot encoding, TF-IDF, và Word Embeddings (Word2Vec, FastText,...).

## 4. Modelling
### 4.1. Baseline Models
These are the baseline models that we used:
- FCNN
- CNN
- LSTM
- Transformers
### 4.2. Optimizing the best baseline model
Among the aforementioned models, CNN's output has the highest accuracy. Therefore, we used CNN to continue to optimize.
Our optimization leanred and applied from the [post](https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist).

### 4.3. Advanced model
We used [AutoGluon for text](https://auto.gluon.ai/stable/tutorials/multimodal/text_prediction/beginner_text.html) to help us find the best model.
With the best model from AutoGluon, we continued to test it with different types of data, and it successfully outplays any other models. 

## 5. Model Results

## 6. Conclusion
______ successfully optimized performance, achieving higher accuracy than traditional models. This project can be expanded by collecting more data or deploying model and making it a real online tool.





