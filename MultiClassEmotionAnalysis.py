import csv
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def getData():
    X = []
    y = []
    c=0
    f = open('C:\\Users\\sagar_000\\Desktop\\BE project\\Multiclass Emotion analysis using NLP\\dataset.csv', 'r')
    reader = csv.reader(f)
    
    for row in reader:
        X.append(row[0])

        if row[1] == "joy":
            y.append(0)
        elif row[1] == "anger":
            y.append(1)
        elif row[1] == "sadness":
            y.append(2)
        elif row[1] == "fear":
            y.append(3)
        elif row[1] == "neutral":
            y.append(4)
        elif row[1] == "love":
            y.append(5)
        else:
            # I am getting 1 row less in label column
            y.append(-1)

    classes = [0, 1, 2, 3, 4, 5]

    X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test


def process(X_train, X_test):
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')

    new_X_train = []
    for item in X_train:
        lemma_list = []
        for word in word_tokenize(item):
            if len(word) <= 1:
                continue
            if word in stop_words:
                continue
            #[line.decode('utf-8').strip() for line in word.readlines()]
            lemma_list.append(lemmatizer.lemmatize(word))
        new_X_train.append(" ".join(lemma_list))

    new_X_test = []
    for item in X_test:
        lemma_list = []
        for word in word_tokenize(item):
            if len(word) <= 1:
                continue
            if word in stop_words:
                continue
            lemma_list.append(lemmatizer.lemmatize(word))
        new_X_test.append(" ".join(lemma_list))
    return new_X_train, new_X_test


def classifier(X_train, y_train):
    # Feature Extraction
    vec = TfidfVectorizer(min_df=0.0, max_df=0.95, use_idf=True, ngram_range=(1, 2))

    # Linear Support Vector Classification
    svm_clf = svm.LinearSVC(C=0.1)
    vec_clf = Pipeline([('vectorizer', vec), ('pac', svm_clf)])
    vec_clf.fit(X_train, y_train)
    joblib.dump(vec_clf, 'svm.pkl', compress=3)
    return vec_clf


def main():
    print("Getting Training Data")
    X_train, X_test, y_train, y_test = getData()
    print("Processing...")
    X_train, X_test = process(X_train, X_test)
    print("Training...")
    vec_clf = classifier(X_train, y_train)
    y_pred = vec_clf.predict(X_test)
    print(sklearn.metrics.classification_report(y_test, y_pred))
    print("Done")


if __name__ == "__main__":
    main()