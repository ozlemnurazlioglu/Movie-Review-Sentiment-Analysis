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
print(df.dropna(inplace=True))
print(len(pos_reviews))


df.to_csv("moviereviews.csv", index=False)

print(df.isnull().sum())




blanks=[]

#(index,label,review text)
for i,lb,rv in df.itertuples():
    if rv.isspace():
        blanks.append(i)
print(blanks)
df.drop(blanks,inplace=True)
print(df['label'].value_counts())


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
df['scores'] = df['review'].apply(lambda review: sid.polarity_scores(review))

df['compound']  = df['scores'].apply(lambda score_dict: score_dict['compound'])

df['comp_score'] = df['compound'].apply(lambda c: 'pos' if c >=0 else 'neg')

df.head()
accuracy_score(df['label'],df['comp_score'])
print(classification_report(df['label'],df['comp_score']))