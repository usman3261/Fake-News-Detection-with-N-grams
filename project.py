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
##print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

pipe=Pipeline([
    ('cv', CountVectorizer(ngram_range=(1, 3))),
    ('nb', MultinomialNB())
    ])
pipe.fit(X_train, y_train) 
y_pred = pipe.predict(X_test)
##print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

pipe=Pipeline([
    ('cv', CountVectorizer(ngram_range=(1, 4))),
    ('nb', MultinomialNB())
    ])
pipe.fit(X_train, y_train) 
y_pred = pipe.predict(X_test)
##print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

test=[
    "Aliens have taken over the White House, reports claim.",
    "New research shows that eating chocolate daily makes you immortal.",
    "Invisible man caught on camera walking in a busy street.",
    "Scientists discover portal to another dimension in the Pacific Ocean.",
    "Man claims to have lived without sleeping for 20 years.",
    "Dinosaurs cloned successfully in underground lab, scientists reveal.",
    "World leaders meet secretly to plan human colonization of Mars next year.",
    "Technology company invents teleportation device for public use.",
    "Mermaids spotted off the coast of California by fishermen.",
    "Drinking coffee cures all types of cancer, says new study.",

   
    "The national cricket team won the international championship yesterday.",
    "Government introduces new tax policy for small businesses.",
    "City hospital opens new wing for cancer treatment patients.",
    "University announces 500 new scholarships for engineering students.",
    "Bridge connecting two major cities opens for traffic next Monday.",
    "Local school wins national robotics competition.",
    "Weather department forecasts heavy rains for the coming week.",
    "Company reports record profits for the third quarter.",
    "National team wins gold medal in international sports event.",
    "Healthcare system introduces new policies for patient care."

]
res=pipe.predict(test)

print("Predictions for test cases:")
for i, text in enumerate(test):
    print(f"Text: {text}\nPrediction: {'Real' if res[i] == 1 else 'Fake'}\n")
