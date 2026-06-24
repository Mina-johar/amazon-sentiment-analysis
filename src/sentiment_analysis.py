import os
import gzip
import json
import pandas as pd
import matplotlib.pyplot as plt
import nltk

from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

os.makedirs("report", exist_ok=True)

bewertungen = []

dateien = {
    "Automotive": "data/Automotive.jsonl.gz",
    "Pet Supplies": "data/Pet_Supplies.jsonl.gz",
    "Video Games": "data/Video_Games.jsonl.gz"
}

for kategorie, datei in dateien.items():

    with gzip.open(
        datei,
        "rt",
        encoding="utf-8"
    ) as f:

        for i, zeile in enumerate(f):

            if i >= 5000:
                break

            daten = json.loads(zeile)

            if (
                "text" in daten
                and
                "rating" in daten
            ):

                titel = daten.get(
                    "title",
                    daten.get("summary", "")
                )

                text = daten.get(
                    "text",
                    ""
                )

                bewertungen.append({
                    "category": kategorie,
                    "text": str(titel) + " " + str(text),
                    "rating": int(daten["rating"])
                })

df = pd.DataFrame(bewertungen)

print("\nAnzahl Datensätze:")
print(len(df))
print("\nKategorien:")

kategorie_tabelle = pd.DataFrame(
    df["category"].value_counts()
)

kategorie_tabelle.columns = ["Anzahl"]

print(kategorie_tabelle)

kategorie_tabelle.to_csv(
    "report/kategorien.csv"
)

rating_tabelle = pd.DataFrame(
    df["rating"].value_counts().sort_index()
)

rating_tabelle.columns = ["Anzahl"]

rating_tabelle["Prozent"] = round(
    rating_tabelle["Anzahl"]
    /
    rating_tabelle["Anzahl"].sum()
    * 100,
    2
)

rating_tabelle.index.name = "Sterne"

print("\nVerteilung Sternebewertungen:")
print(rating_tabelle)

rating_tabelle.to_csv(
    "report/sternbewertungstabelle.csv"
)
nltk.download("wordnet")
nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):

    words = str(text).split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)

df["clean_text"] = df["text"].apply(
    lemmatize_text
)

X = df["clean_text"]
y = df["rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
train_test_tabelle = pd.DataFrame({
    "Training":
        y_train.value_counts().sort_index(),
    "Test":
        y_test.value_counts().sort_index()
})

print(train_test_tabelle)

train_test_tabelle.to_csv(
    "report/train_test_verteilung.csv"
)
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1,2)
)

X_train_vec = vectorizer.fit_transform(
    X_train
)

X_test_vec = vectorizer.transform(
    X_test
)

# Logistic Regression

log_model = LogisticRegression(
    max_iter=1000
)

log_model.fit(
    X_train_vec,
    y_train
)

log_pred = log_model.predict(
    X_test_vec
)

log_accuracy = accuracy_score(
    y_test,
    log_pred
)

# Naive Bayes

nb_model = MultinomialNB()

nb_model.fit(
    X_train_vec,
    y_train
)

nb_pred = nb_model.predict(
    X_test_vec
)

nb_accuracy = accuracy_score(
    y_test,
    nb_pred
)

# SVM

svm_model = LinearSVC()

svm_model.fit(
    X_train_vec,
    y_train
)

svm_pred = svm_model.predict(
    X_test_vec
)

svm_accuracy = accuracy_score(
    y_test,
    svm_pred
)
print(classification_report(
    y_test,
    log_pred
))

print(classification_report(
    y_test,
    nb_pred
))

print(classification_report(
    y_test,
    svm_pred
))

ergebnisse = pd.DataFrame({

    "Modell": [
        "Logistic Regression",
        "Naive Bayes",
        "SVM"
    ],

    "Accuracy (%)": [

        round(
            log_accuracy*100,
            2
        ),

        round(
            nb_accuracy*100,
            2
        ),

        round(
            svm_accuracy*100,
            2
        )
    ]
})

ergebnisse = ergebnisse.sort_values(
    by="Accuracy (%)",
    ascending=False
)

print(ergebnisse)

ergebnisse.to_csv(
    "report/modellvergleich.csv",
    index=False
)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    log_pred
)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    log_pred
)

cm_df = pd.DataFrame(
    cm,
    index=[1,2,3,4,5],
    columns=[1,2,3,4,5]
)

cm_df.to_csv(
    "report/confusion_matrix.csv"
)

plt.title(
    "Confusion Matrix Logistic Regression"
)

plt.savefig(
    "report/confusion_matrix_logistic.png"
)

plt.close()
print("\nAccuracy pro Kategorie")

for kategorie in df["category"].unique():

    subset = df[
        df["category"] == kategorie
    ]

    X_cat = vectorizer.transform(
        subset["clean_text"]
    )

    y_cat = subset["rating"]

    pred_cat = log_model.predict(
        X_cat
    )

    acc_cat = accuracy_score(
        y_cat,
        pred_cat
    )

    print(
        f"{kategorie}: {acc_cat:.4f}"
    )

df["sentiment"] = df["rating"].apply(

    lambda x:

    "negativ"
    if x <= 2

    else "neutral"
    if x == 3

    else "positiv"
)

print("\nSentiment-Verteilung")

print(
    df["sentiment"].value_counts()
)

plt.figure(figsize=(8,5))

df["sentiment"].value_counts().plot(
    kind="bar"
)

plt.title(
    "Sentiment-Verteilung"
)

plt.tight_layout()

plt.savefig(
    "report/sentiment_verteilung.png"
)

plt.close()

print("""
      beste_accuracy = max(
    log_accuracy,
    nb_accuracy,
    svm_accuracy
)

if beste_accuracy == log_accuracy:
    bestes_modell = "Logistic Regression"

elif beste_accuracy == nb_accuracy:
    bestes_modell = "Naive Bayes"

else:
    bestes_modell = "SVM"

print(
    f"\nBestes Modell: {bestes_modell}"
)
==================================================
FAZIT
==================================================

Datensatz:
15.000 Amazon Rezensionen

Kategorien:
- Automotive
- Pet Supplies
- Video Games

Verwendete Modelle:
- Logistic Regression
- Bestes Modell:
      {bestes_modell}
- Naive Bayes
- SVM

Bewertung:
1 bis 5 Sterne

Berechnet:
- Accuracy
- Classification Report
- Confusion Matrix

Zusätzlich:
- Accuracy pro Kategorie
- Sentimentanalyse

Alle Ergebnisse wurden
im Ordner report gespeichert.
==================================================
""")
