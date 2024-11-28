import pandas as pd
pd.options.plotting.backend = "plotly"

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.utils import resample


# PATH = "./fake_emails"
# folders = os.listdir(PATH)
# data = []

# for folder in folders:
#     files = os.listdir(os.path.join(PATH, folder))
#     for file in files:
#         try:
#             with open(os.path.join(PATH, folder,file)) as f:
#                 contents = " ".join(f.readlines())
#                 data.append([file.split(".")[0], folder, contents])
#                 f.close()
#         except Exception as e:
#             pass

# df = pd.DataFrame(data, columns=['ID', 'Category', 'Content'])
# df.to_csv("./combine.csv", index = False)
df = pd.read_csv("./cleaned_reviews.csv")
df = df[df['sentiments'] != 'neutral']

en_stopwords = stopwords.words('english')
word_lemmatizer = WordNetLemmatizer()

def clean_review(contents):
    if not isinstance(contents, str):
        contents = ""
    tokenized_words = word_tokenize(contents.lower())
    clean_words = [
        word_lemmatizer.lemmatize(word) for word in tokenized_words 
        if word not in en_stopwords and word.isalnum() and len(word) > 2
    ]
    return " ".join(clean_words)
df['cleaned_review'] = df['cleaned_review'].apply(clean_review)

# Encode sentiments
le = LabelEncoder()
df['sentiments_encoded'] = le.fit_transform(df['sentiments'])

negative = df[df['sentiments'] == 'negative']
positive = df[df['sentiments'] == 'positive']
negative_upsampled = resample(
    negative,
    replace=True,
    n_samples=len(positive),
    random_state=42
)
df_balanced = pd.concat([positive, negative_upsampled])
# Feature extraction: TF-IDF
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df_balanced['cleaned_review'])
y = df_balanced['sentiments_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
report = classification_report(y_test, predictions, target_names=le.classes_)
print(report)