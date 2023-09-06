# AI-Human_Generated_Text-_Identification
This Machine Learning Model identifies the whether given text is AI genetrated or Human generated with accuracy of 95.3%.


# TF-IDF

Term frequency-inverse document frequency is a text vectorizer that transforms the text into a usable vector. It combines 2 concepts, Term Frequency (TF) and Document Frequency (DF).

The term frequency is the number of occurrences of a specific term in a document. Term frequency indicates how important a specific term is in a document. Term frequency represents every text from the data as a matrix whose rows are the number of documents and columns are the number of distinct terms throughout all documents.

Document frequency is the number of documents containing a specific term. Document frequency indicates how common the term is.

Inverse document frequency (IDF) is the weight of a term, it aims to reduce the weight of a term if the term’s occurrences are scattered throughout all the documents. IDF can be calculated as follow:
  
![image](https://user-images.githubusercontent.com/128599179/233066883-5fb976f0-4bd5-450e-b419-76881fd340a3.png)

Where idfᵢ is the IDF score for term i, dfᵢ is the number of documents containing term i, and n is the total number of documents. The higher the DF of a term, the lower the IDF for the term. When the number of DF is equal to n which means that the term appears in all documents, the IDF will be zero, since log(1) is zero, when in doubt just put this term in the stop word list because it doesn't provide much information.
	
The TF-IDF score as the name suggests is just a multiplication of the term frequency matrix with its IDF, it can be calculated as follow:
  
![image](https://user-images.githubusercontent.com/128599179/233066928-6e29b8fb-c877-419b-a1bf-004621bd0b7d.png)

Where wᵢⱼ is TF-IDF score for term i in document j, tfᵢⱼ is term frequency for term i in document j, and idfᵢ is IDF score for term i.



# Logistic Regression

Logistic regression (or logit regression) estimates the probability of an event occurring, such as yes or no, based on a given dataset of independent variables. Since the outcome is a probability, the dependent variable is bounded betwee n 0 and 1. 
	
In logistic regression, a logit transformation is applied to the odds-that is, the probability of success divided by the probability of failure. This is also commonly known as the log odds or the natural logarithm of odds, and this logistic function is represented by the following formulas:
 
### Logit(pi) = 1/(1+ exp(-pi))



### ln(pi/(1-pi)) = Beta_0 + Beta_1*X_1 + … + B_k*K_k


In this logistic regression equation, logit(pi) is the dependent or response variable and x is the independent variable. For binary classification, a probability less than .5 will predict 0 while a probability greater than 0 will predict 1.  



# Stopword Removal

Stopword removal is a common text preprocessing step in natural language processing (NLP) that involves removing common words from a text that are unlikely to contain important information or meaning. Stopwords are words that are used frequently in a language, such as `"the,"` `"and,"` `"is,"` `"of,"` and `"in."`

The reason for removing stopwords is to reduce the dimensionality of the text data and improve the efficiency of subsequent processing tasks such as text classification or sentiment analysis. Stopwords are often removed from the text before other preprocessing tasks, such as tokenization, stemming, or lemmatization.

Stopword removal algorithms use a list of predefined stopwords for a given language or domain. These lists can be customized to include or exclude specific words depending on the application. Once the list of stopwords is defined, the algorithm will identify and remove any instances of those words from the text.


# Project steps 
+ ## ***Steps for a machine learning model that can determine if a piece of news was produced by a human or an AI***

### Create and Pre-process data 
Create a dataset of news articles that are labelled as Human and AI. Pre-process the data by cleaning the text, removing stop words, and converting the text into numerical features using techniques like TF-IDF.

```
# Load the dataset
data = pd.read_csv("/content/news.csv")

# Preprocess the data 
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(data['text'])
y = data['label']
```  

### Split the data into training and testing sets
Split the dataset into a training set and a testing set to evaluate the model's performance.

```  
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Train a machine learning model 
Train a machine learning model on the training set. Use logistic regression algorithm for classification.

```
# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
```

### Evaluate the model
Evaluate the model's performance on the testing set. You can use metrics like accuracy, precision, recall, and F1 score.

```
# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
```

### Use the model to predict new data
Once the model is trained, you can use it to predict whether a news article is human-generated or AI-generated.

```
# Use the model to predict new data
new_text = ['Researchers develop new way to produce renewable energy']
new_text_transformed = tfidf.transform(new_text)
prediction = model.predict(new_text_transformed)
print('Prediction:', prediction)
```

In order to pre-process the data, we first import the dataset and used the TF-IDF vectorizer. After separating the data into training and testing sets, we used the training set to build a logistic regression model. We assess the model's performance on the testing set and employ it to determine if a recent news story is generated by humans or artificial intelligence.

