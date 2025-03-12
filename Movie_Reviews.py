import os
import pandas as pd


base_path = "C:/Users/ozlem/PycharmProjects/nlp/moviereviews"
neg_path = os.path.join(base_path, "neg")
pos_path = os.path.join(base_path, "pos")

def load_reviews(folder, label):
    reviews = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            reviews.append((f.read(), label))
    return reviews


neg_reviews = load_reviews(neg_path, "negative")
pos_reviews = load_reviews(pos_path, "positive")

df = pd.DataFrame(neg_reviews + pos_reviews, columns=["review", "label"])


print(df.head())
print(len(neg_reviews))
print(len(pos_reviews))


df.to_csv("moviereviews.csv", index=False)


print(df.isnull().sum())



mystring='hello'
empty=' '
print(mystring.isspace())
print(empty.isspace())
blanks=[]

#(index,label,review text)
for i,lb,rv in df.itertuples():
    if rv.isspace():
        blanks.append(i)
print(blanks)
df.drop(blanks,inplace=True)
print(len(df))

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
X=df['review']
y=df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

text_clf=Pipeline([('tfidf', TfidfVectorizer()),
                   ('clf',LinearSVC())])
text_clf.fit(X_train, y_train)
predictions=text_clf.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))




