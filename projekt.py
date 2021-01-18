# Importowanie bibliotek

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import graphviz
from sklearn.tree import export_graphviz

df = pd.read_csv("StudentsPerformance.csv")
dfcopy = df.copy(deep=True)
names_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'math score', 'reading score', 'writing score']
names_class = ['test preparation course']
scoreTest = ["none", "completed"]

print(f"\nLiczba danych: {dfcopy.shape[0]} Liczba kolumn: {dfcopy.shape[1]}")
print("\nSprawdzam czy są jakieś braki w danych")
print(dfcopy.isnull().any())
print("\nSprawdzam sumę błędów w każdej tabeli:")
print(dfcopy.isnull().sum())
print(f'Suma błędów: {df.isnull().sum().sum()}')
print("\nSprawdzam rodzaj danych kolejno w kolumnach")
print(dfcopy.dtypes)


print("\nW danej bazie kolumny posiadają następujące dane:\n")
print(f"Płeć: {dfcopy['gender'].unique()}")
print(f"Przynależność etniczna/rasa: {dfcopy['race/ethnicity'].unique()}")
print(
    f"Wykształcenie rodziców: {dfcopy['parental level of education'].unique()}")
print(f"Czy student płaci za posiłki w uczelni: {dfcopy['lunch'].unique()}")
print(
    f"Czy ukończył kurs przygotowujący: {dfcopy['test preparation course'].unique()}")
print(f"Wyniki egzaminów z matematyki w przediale do 100pkt.")
print(f"Wyniki egzaminów z czytania w przediale do 100pkt.")
print(f"Wyniki egzaminów z pisania w przediale do 100pkt.\n")

print()
print(dfcopy.head(10))
print()

# Zamiana danych obiektowych na dane liczbowe
print("Zamiana danych obietowych na liczbowe...")

le = preprocessing.LabelEncoder()
# płeć
dfcopy['gender'] = le.fit_transform(dfcopy['gender'].values)
# rasa
dfcopy['race/ethnicity'] = le.fit_transform(dfcopy['race/ethnicity'].values)
# wykształcenie rodziców
dfcopy['parental level of education'] = le.fit_transform(
    dfcopy['parental level of education'].values)
# rodzaj posiłków płatne/darmowe
dfcopy['lunch'] = le.fit_transform(dfcopy['lunch'].values)
# egzamin przygotowujący
#dfcopy['test preparation course'] = le.fit_transform(
    #dfcopy['test preparation course'].values)

print()
print(dfcopy.head(10))
print()
print(dfcopy.dtypes)

print("\nSprawdzam odchylenie, kwantyle, max, min itp")
print(dfcopy.describe().T)

# print("""gender: female=0, male=1
# race/ethnicity: 'group A'=0, 'group B'=1, 'group C'=2, 'group D'=3, 'group E'=4
# parental level of education: associate's degree=0, bachelor's degree=1, high school=2,
# master's degree=3, some college=4, some high school=5
# lunch: free/reduced=0, standard=1,
# test preparation course: completed=0, none=1,
# """)

# Obrazuje dane

#value1 = list(df['lunch'].values).count('free/reduced')
#value2 = list(df['lunch'].values).count('standard')

#slices = [value1, value2]
#live = ['Darmowe', 'Płatne',]
#kolory = ['orange','m']

#plt.pie(slices, labels = live, colors=kolory, startangle=90, shadow=True, explode=(0,0.1), autopct='%1.1f%%')

#plt.title("Rodzaj posiłków na stołówce\n")
#plt.show()

#Zalezność między wynikami testów a innymi parametrami obiektowymi
#sns.pairplot(df, hue='test preparation course')
#plt.show()

# Zbadanie zależności między danymi a wynikami z poszczegolnych testów

# KNN

x = np.array(dfcopy[names_features].values)
y = np.array(df['test preparation course'].values)

(train_x, test_x, train_y, test_y) = train_test_split(
    x, y, train_size=0.70, random_state=1)

# k=3

knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(train_x, train_y)
accuracyKnn3 = knn3.score(test_x, test_y)
predk3 = knn3.predict(test_x)
mKnn3 = confusion_matrix(test_y, predk3)

# k=5

knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(train_x, train_y)
accuracyKnn5 = knn5.score(test_x, test_y)
predk5 = knn5.predict(test_x)
mKnn5 = confusion_matrix(test_y, predk5)

# k=11

knn11 = KNeighborsClassifier(n_neighbors=11)
knn11.fit(train_x, train_y)
accuracyKnn11 = knn11.score(test_x, test_y)
predk11 = knn11.predict(test_x)
mKnn11 = confusion_matrix(test_y, predk11)

# Naive-Bayes

modelNb = GaussianNB()
modelTrainNb = modelNb.fit(train_x, train_y)
predNb = modelNb.predict(test_x)
mNb = confusion_matrix(test_y, predNb)
accuracyNb = modelNb.score(test_x, test_y)

# Support vector machine SVM

modelSVM = SVC()
modelSVM.fit(train_x,train_y)
predSVM = modelSVM.predict(test_x)
mSVM = confusion_matrix(test_y, predSVM)
accuracySVM = modelSVM.score(test_x, test_y)

# Random forest

modelRF = RandomForestClassifier()
modelRF.fit(train_x,train_y)
predRF = modelRF.predict(test_x)
mRF = confusion_matrix(test_y, predRF)
accuracyRF = modelRF.score(test_x, test_y)

# Decision tree with bagging
bg = BaggingClassifier(DecisionTreeClassifier())
bgTrain = bg.fit(train_x, train_y)
predbg = bgTrain.predict(test_x)
mbg = confusion_matrix(test_y, predbg)
accuracybg = bg.score(test_x, test_y)


# Decision tree

modelTree = DecisionTreeClassifier()
modelTrainTree = modelTree.fit(train_x, train_y)
predTree = modelTree.predict(test_x)
mTree = confusion_matrix(test_y, predTree)
accuracyTree = modelTree.score(test_x, test_y)

# Wygenerowanie drzewa decyzyjnego
#dot_data = export_graphviz(modelTree, out_file=None, feature_names=names_features, class_names=scoreTest, filled=True, rounded=True, special_characters=True)
#graph = graphviz.Source(dot_data)
#graph.render("projectTest")


# siec neuronowa

# Skalowanie danych

scaler = StandardScaler()

# Dopasowanie do danych treningowych

scaler.fit(train_x)

# Skalowanie danych treningowych

train_data = scaler.transform(train_x)
test_data = scaler.transform(test_x)


# Stworzenie klasyfikatora

mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=10000)

# Dopasowanie danych treningowych do modelu

mlp.fit(train_x, train_y)
predictions_train = mlp.predict(train_x)
predictions_test = mlp.predict(test_x)
accuracyNeural_network = accuracy_score(predictions_test, test_y)


# Wyniki

print("\nEwaluacja algorytmów:")
print(f"KNN, k = 3: {accuracyKnn3}")
print("Macierz błędu:")
print(mKnn3)
print(f"KNN, k = 5: {accuracyKnn5}")
print("Macierz błędu:")
print(mKnn5)
print(f"KNN, k = 11: {accuracyKnn11}")
print("Macierz błędu:")
print(mKnn11)
print(f"Naive-Bayes: {accuracyNb}")
print("Macierz błędu:")
print(mNb)
print(f"Support vector machine SVM: {accuracySVM}")
print("Macierz błędu:")
print(mSVM)
print(f"Random forest: {accuracyRF}")
print("Macierz błędu:")
print(mRF)
print(f"Decision tree: {accuracyTree}")
print("Macierz błędu:")
print(mTree)
print(f"Decision tree with bagging: {accuracybg}")
print("Macierz błędu:")
print(mbg)
print(f"Neural network: {accuracyNeural_network}")
print("Macierz błędu:")
print(confusion_matrix(predictions_test, test_y))

the_bestAlg = max(accuracyKnn3, accuracyKnn5, accuracyKnn11, accuracyNb, accuracySVM, accuracyTree, accuracybg, accuracyNeural_network)

if the_bestAlg == accuracyKnn3:
    print(f"\nW tym przypadku najlepszym algorytmem jest KNN, k=3: {accuracyKnn3}")
    print("\nWyniki ogólne:")
    print(classification_report(predk3, test_y))
elif the_bestAlg == accuracyKnn5:
    print(f"\nW tym przypadku najlepszym algorytmem jest KNN, k=5: {accuracyKnn5}")
    print("\nWyniki ogólne:")
    print(classification_report(predk5, test_y))
elif the_bestAlg == accuracyKnn11:
    print(f"\nW tym przypadku najlepszym algorytmem jest KNN, k=11: {accuracyKnn11}")
    print("\nWyniki ogólne:")
    print(classification_report(predk11, test_y))
elif the_bestAlg == accuracyNb:
    print(f"\nW tym przypadku najlepszym algorytmem jest Naive-Bayes: {accuracyNb}")
    print("\nWyniki ogólne:")
    print(classification_report(predNb, test_y))
elif the_bestAlg == accuracySVM:
    print(f"\nW tym przypadku najlepszym algorytmem jest Support vector machine SVM: {accuracySVM}")
    print("\nWyniki ogólne:")
    print(classification_report(predSVM, test_y))
elif the_bestAlg == accuracyRF:
    print(f"\nW tym przypadku najlepszym algorytmem jest Random Forest: {accuracyRF}")
    print("\nWyniki ogólne:")
    print(classification_report(predRF, test_y))
elif the_bestAlg == accuracyTree:
    print(f"\nW tym przypadku najlepszym algorytmem jest Decision tree: {accuracyTree}")
    print("\nWyniki ogólne:")
    print(classification_report(predTree, test_y))
elif the_bestAlg == accuracybg:
    print(f"\nW tym przypadku najlepszym algorytmem jest Decision tree with bagging: {accuracybg}")
    print("\nWyniki ogólne:")
    print(classification_report(predbg, test_y))
elif the_bestAlg == accuracyNeural_network:
    print(f"\nW tym przypadku najlepszym algorytmem jest Neural network: {accuracyNeural_network}")
    print("\nWyniki ogólne:")
    print(classification_report(predictions_test, test_y))

# Wykres słupkowy wyników

plt.figure(figsize=(11, 10))

# Create bars
plt.bar(np.arange(9), [accuracyKnn3, accuracyKnn5, accuracyKnn11,accuracyNb, accuracySVM, accuracyRF, accuracyTree, accuracybg, accuracyNeural_network], color='#969696')

# Create names on the x-axis
plt.xticks(np.arange(9), ['Knn3', 'Knn5', 'Knn11', 'Native-Bayes', 'SVM', 'Random forest', 'Decision tree', 'Bagging Decision Tree', 'Neural network'])

plt.xlabel('Algorytm ewaluacji', fontsize=12, color='#323232')
plt.ylabel('Poziom skuteczności (%)', fontsize=12, color='#323232')
plt.title('Wyniki', fontsize=16, color='#323232')

# Show graphic
plt.show()
