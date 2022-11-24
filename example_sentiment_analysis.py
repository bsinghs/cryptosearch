import pandas as pd


df = pd.read_csv('Reviews.csv')
print(df.head())

# Now, we will take a look at the variable “Score” to see if 
# majority of the customer ratings are positive or negative.
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
#matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
# Product Scores
# fig = px.histogram(df, x="Score")
# fig.update_traces(marker_color="turquoise",marker_line_color='rgb(8,48,107)',
#                   marker_line_width=1.5)
# fig.update_layout(title_text='Product Score')
#fig.show()

#NOT WORKING
# Now, we can create some wordclouds to see the most frequently used words in the reviews.
# import nltk
# from nltk.corpus import stopwords
# # Create stopword list:
# stopwords = set(STOPWORDS)
# stopwords.update(["br", "href"])
# textt = " ".join(review for review in df.Text)
# wordcloud = WordCloud(stopwords=stopwords).generate(textt)
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.savefig('wordcloud11.png')
# plt.show()

#We will classify all reviews with ‘Score’ > 3 as +1, indicating that they are positive.
#All reviews with ‘Score’ < 3 will be classified as -1. Reviews with ‘Score’ = 3 will be dropped
# assign reviews with score > 3 as positive sentiment
# score < 3 negative sentiment
# remove score = 3
df = df[df['Score'] != 3]
df['sentiment'] = df['Score'].apply(lambda rating : +1 if rating > 3 else -1)

#Finally, we can take a look at the distribution of reviews with sentiment across the dataset:
# df['sentimentt'] = df['sentiment'].replace({-1 : 'negative'})
# df['sentimentt'] = df['sentimentt'].replace({1 : 'positive'})
# fig = px.histogram(df, x="sentimentt")
# fig.update_traces(marker_color="indianred",marker_line_color='rgb(8,48,107)',
#                   marker_line_width=1.5)
# fig.update_layout(title_text='Product Sentiment')
# fig.show()


def remove_punctuation(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":",  "!",'"'))
    return final

df['Text'] = df['Text'].apply(remove_punctuation)
df = df.dropna(subset=['Summary'])
df['Summary'] = df['Summary'].apply(remove_punctuation)

dfNew = df[['Summary','sentiment']]
print(dfNew.head())


# random split train and test data
import numpy as np
index = df.index
df['random_number'] = np.random.randn(len(index))
train = df[df['random_number'] <= 0.8]
test = df[df['random_number'] > 0.8]

#We will need to convert the text into a bag-of-words model 
#since the logistic regression algorithm cannot understand text.
# count vectorizer:
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train['Summary'])
test_matrix = vectorizer.transform(test['Summary'])

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

#Split target and independent variables
X_train = train_matrix
X_test = test_matrix
y_train = train['sentiment']
y_test = test['sentiment']

#Fit model on data
lr.fit(X_train,y_train)
#Make predictions
predictions = lr.predict(X_test)

# find accuracy, precision, recall:
from sklearn.metrics import confusion_matrix,classification_report
new = np.asarray(y_test)
confusion_matrix(predictions,y_test)