import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

df=pd.read_csv('fake_news.csv')
df['label_num']=df['label'].map({'Fake': 0, 'Real': 1})

X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['label_num'], test_size=0.2, random_state=42,stratify=df['label_num'])

pipe=Pipeline([
    ('cv', CountVectorizer(ngram_range=(1, 2))),
    ('nb', MultinomialNB())
    ])
pipe.fit(X_train, y_train) 
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

pipe=Pipeline([
    ('cv', CountVectorizer(ngram_range=(1, 3))),
    ('nb', MultinomialNB())
    ])
pipe.fit(X_train, y_train) 
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

pipe=Pipeline([
    ('cv', CountVectorizer(ngram_range=(1, 4))),
    ('nb', MultinomialNB())
    ])
pipe.fit(X_train, y_train) 
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))