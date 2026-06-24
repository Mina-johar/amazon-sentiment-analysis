# amazon-sentiment-analysis
Sentimentanalysis von Amazon Produktrezensionen mit Logistic Regression, SVM

## Projektbeschreibung

Dieses Projekt entwickelt ein Textklassifikationssystem zur automatischen Bewertung von Amazon-Produktrezensionen. Ziel ist die Vorhersage der numerischen Sternebewertung (1 bis 5 Sterne) anhand des Rezensionstextes.

## Datensatz

Verwendet wurde der öffentliche Datensatz Amazon Reviews 2023 (McAuley Lab, 2023).

Berücksichtigte Produktkategorien:

* Automotive
* Pet Supplies
* Video Games

Insgesamt wurden 15.000 Rezensionen analysiert.

## Datenvorverarbeitung

Die Rezensionstexte wurden vorverarbeitet durch:

* Tokenisierung
* Entfernen von Stoppwörtern
* Lemmatisierung
* TF-IDF-Vektorisierung

Anschließend wurden die Daten in Trainings- und Testdaten aufgeteilt.

## Verwendete Modelle

Folgende Klassifikationsmodelle wurden verglichen:

* Logistic Regression
* Support Vector Machine (SVM)
* Naive Bayes

## Ergebnisse

| Modell              | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 70.37 %  |
| SVM                 | 69.17 %  |
| Naive Bayes         | 66.57 %  |

Logistic Regression erzielte die beste Genauigkeit und wurde daher als Hauptmodell verwendet.

## Visualisierungen

Das Repository enthält:

* Verteilung der Bewertungen
* Sentiment-Verteilung
* Confusion Matrix
* Modellvergleich
* Trainings-/Testdatenverteilung

## Repository-Inhalt

- README.md – Projektbeschreibung
- src/sentiment_analysis.py – Hauptprogramm
- images/ – Visualisierungen
- results/ – Auswertungstabellen und Ergebnisse

## Autor

Mina Johar

