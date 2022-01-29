# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 12:05:05 2021

@author: Aleksandra Mitro
"""

#%% Importovanje biblioteka
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Ucitavanje baze podataka
df = pd.read_csv("recipes.csv")

#%% U bazi se nalazi 10566 recepata i za svaki je zabelezeno prisustvo/odsustvo 150 razlicitih sastojaka
# i poreklo recepata
print(df.shape)
#%% Izbacujemo nepotrebno obelezje Unnamed koje nam ne daje nikakve informacije
df.drop('Unnamed: 0', inplace=True, axis=1)
#%% Provera nedostajajucih vrednosti
print(df.isna().sum().sum())
# U bazi se ne nalazi nijedna nedostajajuca vrednost
#%% Provera ispravnosti numerickih (bool) podataka
dfDescribe = df.describe()
# U bazi se nalaze ispravne vrednosti
#%%
countries = df.iloc[:,-1]
#%% Drzave/Regioni iz kojih recepti poticu
print(df["country"].unique())
# Dakle recepti su iz Juzne Amerike, Francuske, Grcke, Meksika, Italije, 
# Japana, Kine, Tajlanda i Velike Britanije
#%% Preimenovanje kolone kako bi mogla uraditi reset_index nad groupBy funkcijom
df.rename(columns = {"country" : "origin"},inplace = True)
#%% Grupisanje recepata po drzavama/regionima
countryGby = df.groupby(by = countries).count().reset_index()
#%% Crtanje barplota za prikaz broja recepata po poreklu
plt.barh(countryGby["country"],countryGby["origin"])
plt.xlabel("Broj recepata")
plt.ylabel("Poreklo recepata")
plt.title("Prikaz broja recepata po poreklu")
#%% Grupisanje sastojaka recepata po drzavama/regionima
ingredientsGby = df.groupby(by = countries).sum().reset_index()
#%% 
countryUnique = df["origin"].unique()
countryUni = pd.DataFrame(countryUnique)
#%% Najkorisceniji sastojci i najmanje korisceni za svaki od regiona 
maxIng = ingredientsGby.select_dtypes(np.float64).idxmax(axis = 1)
minIng = ingredientsGby.select_dtypes(np.float64).idxmin(axis = 1)
#%% Spajanje 

ingMaxMin = pd.concat([countryUni,maxIng,minIng], axis = 1)
# So je najcesce koriscen sastojak u juznoamerickoj, grckoj, kinekoj i tajlandskoj hrani
# Ocekivano je ulje najcesce koriscen sastojak u italiljanskoj kuhinji kao i meksickoj
# A sos je najcesce koriscen sastojak u francuskoj i britanskoj kuhinji
# Riblje ulje je najredje koriscen sastojak u grckoj i italijanskoj kuhinji
#%% Sumiranje koriscenja sastojaka u receptima

sumIngredients = df.sum(axis = 0).reset_index()
# Najcesce korisceni sastojci su so, ulje, luk, biber i secer
#%% 5 najcesce koriscenih sastojaka
fiveMostUsed = sumIngredients[0:5]
#%% Prikaz 5 najcesce koriscenih sastojaka 
plt.barh(fiveMostUsed['index'],fiveMostUsed[0])
plt.xlabel('Broj recepata')
plt.ylabel('Sastojak')
#%% Prosecan broj sastojaka po receptu
sumOfAllIngredients = df.sum()
sumOfAllIngredients.drop('origin', axis = 0, inplace = True)
#%%
totalSum = sumOfAllIngredients.sum(axis = 0)
#%% Prosecna kolicina sastojaka po receptu
avgNumOfIngredients = totalSum/10566
print(round(avgNumOfIngredients,2))
# 12.01 sastojak se nalazi u svakom receptu
#%% 
cols = df.columns
#%% Izdvanjanje jela koji imaju meso kao sastojak
mealsWithMeat = df[['origin','boneless skinless chicken','bacon','skinless chicken','skinless chicken breasts','boneless skinless chicken breasts','ground beef','fillets']]
#%% Cuvamo samo ona jela gde se koristi meso
withMeat = mealsWithMeat[(mealsWithMeat['boneless skinless chicken']!=0)|(mealsWithMeat['bacon']!=0)|(mealsWithMeat['skinless chicken']!=0)|(mealsWithMeat['skinless chicken breasts']!=0)|
                         (mealsWithMeat['boneless skinless chicken breasts']!=0)|(mealsWithMeat['ground beef']!=0)|(mealsWithMeat['fillets']!=0)]
#1559 recepata sadrzi meso 
#%% Procenat recepata koji sadrze meso
print(1559/10566*100)
#%% Grupisanje recepata po drzavama 
countryMeatGby = withMeat.groupby(by = countries).count().reset_index()
#%% Crtanje barplota za prikaz broja recepata po poreklu
plt.barh(countryMeatGby["country"],countryMeatGby["origin"]/countryGby['origin']*100)
plt.xlabel("Broj recepata")
plt.ylabel("Poreklo recepata")
plt.title("Procenat recepata po poreklu koji sadrze meso")
#%%
column_names = ['boneless skinless chicken', 'boneless skinless chicken breasts', 'skinless chicken', 'skinless chicken breasts','fillets']

withMeat['chicken']= withMeat[column_names].sum(axis=1)
#%% Uklanjanje kolona za pojedinacne vrste piletine
withMeat.drop(column_names,inplace=True, axis=1)
#%% Zamena vrednosti sa 1
withMeat['chicken'].replace([1,2,3,4,5],1,inplace=True)
#%% Najcesce korisceno meso
sumMeatIngredients = withMeat.sum(axis = 0).reset_index()
# najcesce koriscena je piletina zatim slanina a odmah zatim govedina
#%%
chicken = withMeat.drop(['ground beef','bacon'],axis = 1)
#%% 
chickenCountry = chicken.groupby(['origin']).sum().reset_index()
#%% 
withMeat = withMeat.groupby(['origin']).sum().reset_index()
#%%
withMeat.set_index('origin').plot.bar()
#%% --- SVM klasifikator ---

#%% Importovanje biblioteka za SVM klasifikaciju
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
#%%
df = pd.read_csv("recipes.csv")
#%%
df.drop('Unnamed: 0', inplace=True, axis=1)
df.rename(columns = {"country" : "origin"},inplace = True)

#%% Podela data seta na trening i test data set
data, data_test = train_test_split(df, test_size=0.5, random_state=42, shuffle=True)

#%% Izdvajanje data seta na x i y
X = data.iloc[:,:-1].copy()
y = data.iloc[:,-1].copy()
labels_y = y.unique()
print(labels_y)
#%%
def mere_upesnosti(data_test,y_pred):
    print('procenat pogodjenih uzoraka: ', accuracy_score(data_test.iloc[:,-1], y_pred))
    print('preciznost mikro: ', precision_score(data_test.iloc[:,-1], y_pred, average='micro'))
    print('preciznost makro: ', precision_score(data_test.iloc[:,-1], y_pred, average='macro'))
    print('osetljivost mikro: ', recall_score(data_test.iloc[:,-1], y_pred, average='micro'))
    print('osetljivost makro: ', recall_score(data_test.iloc[:,-1], y_pred, average='macro'))
    print('f mera mikro: ', f1_score(data_test.iloc[:,-1], y_pred, average='micro'))
    print('f mera makro: ', f1_score(data_test.iloc[:,-1], y_pred, average='macro'))
#%%
def tacnost_po_klasi(mat_konf, klase):
    tacnost_i = []
    N = mat_konf.shape[0]
    for i in range(N):
        j = np.delete(np.array(range(N)),i) 
        TP = mat_konf[i,i]
        F = 0
        F = (sum(mat_konf[i,j]) + sum(mat_konf[j,i]))
        TN = sum(sum(mat_konf)) - F - TP
        tacnost_i.append((TP+TN)/sum(sum(mat_konf)))
        print('Za klasu ', klase[i], ' tacnost je: ', tacnost_i[i])
    tacnost_avg = np.mean(tacnost_i)
    return tacnost_avg
#%%
def osetljivost_po_klasi(mat_konf, klase):
    osetljivost_i = []
    N = mat_konf.shape[0]
    for i in range(N):
        j = np.delete(np.array(range(N)),i) 
        TP = mat_konf[i,i]
        FN = sum(mat_konf[i,j])
        osetljivost_i.append(TP/(TP+FN))
        print('Za klasu ', klase[i], ' osetljivost je: ', osetljivost_i[i])
    osetljivost_avg = np.mean(osetljivost_i)
    return osetljivost_avg

#%% one versus rest
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
indexes = kf.split(X, y)
acc = []
conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
for train_index, test_index in indexes:
    classifier = SVC(C=100, kernel='rbf', decision_function_shape='ovr')
    classifier.fit(X.iloc[train_index,:].values, y.iloc[train_index])
    y_pred = classifier.predict(X.iloc[test_index,:].values)
    conf_mat += confusion_matrix(y.iloc[test_index], y_pred)
    
    
disp = ConfusionMatrixDisplay(confusion_matrix =conf_mat,  display_labels=classifier.classes_)
disp.plot(cmap="BuGn", values_format='', xticks_rotation=90)  
plt.xlabel('Prediktovane labele')
plt.ylabel('Stvarne labele') 
plt.show()
print('procenat tacno predvidjenih: ', sum(np.diag(conf_mat))/sum(sum(conf_mat)))
mere_uspesnosti(conf_mat)
#%% one versus one
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
indexes = kf.split(X, y)
acc = []
conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
for train_index, test_index in indexes:
    classifier = SVC(C=100, kernel='rbf', decision_function_shape='ovo')
    classifier.fit(X.iloc[train_index,:].values, y.iloc[train_index])
    y_pred = classifier.predict(X.iloc[test_index,:].values)
    conf_mat += confusion_matrix(y.iloc[test_index], y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix =conf_mat,  display_labels=classifier.classes_)
disp.plot(cmap="BuGn", values_format='', xticks_rotation=90)  
plt.xlabel('Prediktovane labele')
plt.ylabel('Stvarne labele') 
plt.show()
print('procenat tacno predvidjenih: ', sum(np.diag(conf_mat))/sum(sum(conf_mat)))
mere_uspesnosti(conf_mat)
#%% sa koriscenjem parametra break_ties
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
indexes = kf.split(X, y)
acc = []
conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
for train_index, test_index in indexes:
    classifier = SVC(C=100, kernel='rbf', decision_function_shape='ovr', break_ties = True)
    classifier.fit(X.iloc[train_index,:].values, y.iloc[train_index])
    y_pred = classifier.predict(X.iloc[test_index,:].values)
    conf_mat += confusion_matrix(y.iloc[test_index], y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix =conf_mat,  display_labels=classifier.classes_)
disp.plot(cmap="BuGn", values_format='', xticks_rotation=90)  
plt.xlabel('Prediktovane labele')
plt.ylabel('Stvarne labele') 
plt.show()
print('procenat tacno predvidjenih: ', sum(np.diag(conf_mat))/sum(sum(conf_mat)))
mere_uspesnosti(conf_mat)
#%% Tacnost i osetljivost za svaku od klasa
print('prosecna tacnost je: ', tacnost_po_klasi(conf_mat, y.unique()))
print('prosecna osetljivost je: ', osetljivost_po_klasi(conf_mat, y.unique()))
#%% Linearni SVM klasifikator
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
indexes = kf.split(X, y)
acc = []
conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
for train_index, test_index in indexes:
    classifier = LinearSVC(C=100, multi_class='ovr', loss='squared_hinge', dual=False)
    classifier.fit(X.iloc[train_index,:].values, y.iloc[train_index])
    y_pred = classifier.predict(X.iloc[test_index,:].values)
    conf_mat += confusion_matrix(y.iloc[test_index], y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix =conf_mat,  display_labels=classifier.classes_)
disp.plot(cmap="BuGn", values_format='', xticks_rotation=90)  
plt.xlabel('Prediktovane labele')
plt.ylabel('Stvarne labele') 
plt.show()
print('procenat tacno predvidjenih: ', sum(np.diag(conf_mat))/sum(sum(conf_mat)))
mere_uspesnosti(conf_mat)
#%% Objedinjeno sa break ties
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
acc = []
for c in [1, 5, 10, 20]:
    for F in ['linear', 'rbf']:
        for mc in ['ovr']:
            indexes = kf.split(X, y)
            acc_tmp = []
            fin_conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
            for train_index, test_index in indexes:
                classifier = SVC(C=c, kernel=F, decision_function_shape=mc, break_ties = True)
                classifier.fit(X.iloc[train_index,:], y.iloc[train_index])
                y_pred = classifier.predict(X.iloc[test_index,:])
                acc_tmp.append(accuracy_score(y.iloc[test_index], y_pred))
                fin_conf_mat += confusion_matrix(y.iloc[test_index], y_pred, labels=labels_y)
            print('za parametre C=', c, ', kernel=', F, ' i pristup ', mc, ' tacnost je: ', np.mean(acc_tmp),
                  ' a mat. konf. je:')

            disp = ConfusionMatrixDisplay(confusion_matrix =fin_conf_mat,  display_labels=classifier.classes_)
            disp.plot(cmap="Blues", values_format='', xticks_rotation=90)  
            plt.show()

            acc.append(np.mean(acc_tmp))
print('najbolja tacnost je u iteraciji broj: ', np.argmax(acc))
#%% Objedinjeno 
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
acc = []
for c in [1, 5, 10, 20]:
    for F in ['linear', 'rbf']:
        for mc in ['ovo', 'ovr']:
            indexes = kf.split(X, y)
            acc_tmp = []
            fin_conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
            for train_index, test_index in indexes:
                classifier = SVC(C=c, kernel=F, decision_function_shape=mc)
                classifier.fit(X.iloc[train_index,:], y.iloc[train_index])
                y_pred = classifier.predict(X.iloc[test_index,:])
                acc_tmp.append(accuracy_score(y.iloc[test_index], y_pred))
                fin_conf_mat += confusion_matrix(y.iloc[test_index], y_pred, labels=labels_y)
            print('za parametre C=', c, ', kernel=', F, ' i pristup ', mc, ' tacnost je: ', np.mean(acc_tmp),
                  ' a mat. konf. je:')

            disp = ConfusionMatrixDisplay(confusion_matrix =fin_conf_mat,  display_labels=classifier.classes_)
            disp.plot(cmap="Blues", values_format='', xticks_rotation=90)  
            plt.show()

            acc.append(np.mean(acc_tmp))
print('najbolja tacnost je u iteraciji broj: ', np.argmax(acc))
#%% Obuka
classifier = SVC(C=1, kernel='rbf', decision_function_shape='ovr')
classifier.fit(X, y)
y_pred = classifier.predict(data_test.iloc[:,:-1])
conf_mat = confusion_matrix(data_test.iloc[:,-1], y_pred, labels=labels_y)

#print(conf_mat)
disp = ConfusionMatrixDisplay(confusion_matrix =conf_mat,  display_labels=classifier.classes_)
disp.plot(cmap="BuGn", values_format='', xticks_rotation=90)  
plt.show()

print('procenat pogodjenih uzoraka: ', accuracy_score(data_test.iloc[:,-1], y_pred))
print('preciznost mikro: ', precision_score(data_test.iloc[:,-1], y_pred, average='micro'))
print('preciznost makro: ', precision_score(data_test.iloc[:,-1], y_pred, average='macro'))
print('osetljivost mikro: ', recall_score(data_test.iloc[:,-1], y_pred, average='micro'))
print('osetljivost makro: ', recall_score(data_test.iloc[:,-1], y_pred, average='macro'))
print('f mera mikro: ', f1_score(data_test.iloc[:,-1], y_pred, average='micro'))
print('f mera makro: ', f1_score(data_test.iloc[:,-1], y_pred, average='macro'))
#%% Importovanje biblioteka za neuronskih mreza
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
#%%
df = pd.read_csv("recipes.csv")
#%%
df.drop('Unnamed: 0', inplace=True, axis=1)
#%%
data, data_test = train_test_split(df, test_size=0.5, random_state=42, shuffle=True)
#%%
dfTemp = pd.read_csv("recipes.csv")
#%% Razdvajanje podataka za predikciju
X = data.iloc[:,:-1].copy()
y = data.iloc[:,-1].copy()
labels_y = y.unique()
print(X.shape)
print(y.unique())
#%%
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df["country"] = le.fit_transform(df["country"])
LabelEncoder()
#%%
X.groupby(by=y).describe()
#%%
def tacnost_po_klasi(mat_konf, klase):
    tacnost_i = []
    N = mat_konf.shape[0]
    for i in range(N):
        j = np.delete(np.array(range(N)),i) 
        TP = mat_konf[i,i]
        F = 0
        F = (sum(mat_konf[i,j]) + sum(mat_konf[j,i]))
        TN = sum(sum(mat_konf)) - F - TP
        tacnost_i.append((TP+TN)/sum(sum(mat_konf)))
        print('Za klasu ', klase[i], ' tacnost je: ', tacnost_i[i])
    tacnost_avg = np.mean(tacnost_i)
    return tacnost_avg
#%%
def osetljivost_po_klasi(mat_konf, klase):
    osetljivost_i = []
    N = mat_konf.shape[0]
    for i in range(N):
        j = np.delete(np.array(range(N)),i) 
        TP = mat_konf[i,i]
        FN = sum(mat_konf[i,j])
        osetljivost_i.append(TP/(TP+FN))
        print('Za klasu ', klase[i], ' osetljivost je: ', osetljivost_i[i])
    osetljivost_avg = np.mean(osetljivost_i)
    return osetljivost_avg
#%%
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
indexes = kf.split(X, y)
acc = []
fin_conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
for train_index, test_index in indexes:
    classifier = MLPClassifier(hidden_layer_sizes=(50,50,50), activation='tanh',
                              solver='adam', batch_size=50, learning_rate='constant', 
                              learning_rate_init=0.001, max_iter=100, shuffle=True,
                              random_state=42, early_stopping=True, n_iter_no_change=10,
                              validation_fraction=0.1, verbose=False)
    classifier.fit(X.iloc[train_index,:].values, y.iloc[train_index])
    y_pred = classifier.predict(X.iloc[test_index,:].values)
    plt.figure
    # iscrtavanje tacnosti validacionog skupa
    plt.plot(classifier.validation_scores_)
    plt.plot(classifier.loss_curve_)
    plt.show()
    print(accuracy_score(y.iloc[test_index], y_pred))
    fin_conf_mat += confusion_matrix(y.iloc[test_index], y_pred)

print('procenat tacno predvidjenih: ', sum(np.diag(fin_conf_mat))/sum(sum(fin_conf_mat)))
disp = ConfusionMatrixDisplay(confusion_matrix =fin_conf_mat,  display_labels=classifier.classes_)
disp.plot(cmap="Blues", values_format='',xticks_rotation=90)
plt.xlabel('Prediktovane labele')
plt.ylabel('Stvarne labele')  
plt.show()
# sa tri skrivena sloja sa po 64 neurona i maksimalnim brojem iteracija 100 procenat tacno predvidjenih je 0.707
# sa pet skrivenih slojeva sa po 64 neurona i maksimalnim brojem iteracija 100 procenat tacno predvidjenih je 0.711
# sa pet skrivenih slojeva sa po 72 neurona i maksimalnim brojem iteracija 100 procenat tacno predvidjenih je 0.705
# sa pet skrivenih slojeva sa po 56 neurona i maksimalnim brojem iteracija 100 procenat tacno predvidjenih je 0.707
# sa pet skrivenih slojeva sa po 64 neurona i maksimalnim brojem iteracija 150 procenat tacno predvidjenih je 0.711
# sa tri skrivena sloja sa po 72 neurona i maks broj iteracija 100 tacno predvidjenih je 0.708
# sa tri skrivena sloja sa po 82 neurona i maks broj iteracija 100 tacno predvidjenih je 0.710
# sa tri skrivena sloja sa po 50 neurona i maks broj iteracija 100 tacno predvidjenih je 0.713
# sa pet skrivenih slojeva sa po 50 neurona i maks broj iteracija 100 tacno predvidjenih 0.711
#%% Tacnost za svaku od klasa
le.inverse_transform(y)
print('prosecna tacnost je: ', tacnost_po_klasi(conf_mat, y.unique()))
print('prosecna osetljivost je: ', osetljivost_po_klasi(conf_mat, y.unique()))

#%% Balansiranje juznoamerickih recepata tako sto izbacujemo svaki drugi uzoraka pomenute klase
SUS_ind = y.loc[y=='southern_us'].index
SUS_ind_red = SUS_ind[::2]
rest_ind = y.loc[y!='southern_us'].index
keep_ind = np.concatenate((SUS_ind_red,rest_ind))
y_undersample = y.copy(deep=True)
X_undersample = X.copy(deep=True)
for i in range(len(y)):
    if i not in keep_ind:
        y_undersample.drop(i, axis=0, inplace=True)
        X_undersample.drop(i, axis=0, inplace=True)
        
print('original X: ', X.shape)
print('original y: ', y.shape)
print('redukovan X: ', X_undersample.shape)
print('redukovan y: ', y_undersample.shape)
y_undersample.groupby(by=y_undersample).count()
#%%
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
indexes = kf.split(X_undersample, y_undersample)
acc = []
fin_conf_mat = np.zeros((len(np.unique(y_undersample)),len(np.unique(y_undersample))))
for train_index, test_index in indexes:
    classifier = MLPClassifier(hidden_layer_sizes=(50,50,50), activation='tanh',
                              solver='adam', batch_size=50, learning_rate='constant', 
                              learning_rate_init=0.001, max_iter=50, shuffle=True,
                              random_state=42, early_stopping=True, n_iter_no_change=10,
                              validation_fraction=0.1, verbose=False)
    classifier.fit(X_undersample.iloc[train_index,:].values, y_undersample.iloc[train_index])
    y_pred = classifier.predict(X_undersample.iloc[test_index,:].values)
    plt.figure
    # iscrtavanje tacnosti validacionog skupa
    plt.plot(classifier.validation_scores_)
    plt.plot(classifier.loss_curve_)
    plt.show()
    print(accuracy_score(y_undersample.iloc[test_index], y_pred))
    fin_conf_mat += confusion_matrix(y_undersample.iloc[test_index], y_pred)

print('konacna matrica konfuzije: \n', fin_conf_mat)
disp = ConfusionMatrixDisplay(confusion_matrix =fin_conf_mat,  display_labels=classifier.classes_)
disp.plot(cmap="BuGn", values_format='',xticks_rotation=90)  
plt.xlabel('Prediktovane labele')
plt.ylabel('Stvarne labele')  
plt.show()

print('procenat tacno predvidjenih: ', sum(np.diag(fin_conf_mat))/sum(sum(fin_conf_mat)))
# sa 3 sloja sa po 64 neurona procenat tacno predvidjenih je 70.66
# sa 5 slojeva sa po 64 neurona procenat tacno predvidjenih je 70.80
# sa 3 sloja sa po 72 neurona procenat tacno predvidjenih je 70.80
# sa 5 slojeva sa po 72 neurona procenat tacno predvidjenih je 70.16
# sa 3 sloja sa po 80 neurona procenat tacno predvidjenih je 70.26
# sa 5 slojeva sa po 80 neurona procenat tacno predvidjenih je 70.28
# sa 3 sloja sa po 50 neurona procenat tacno predvidjenih je 70.31
# sa 5 slojeva sa po 50 neurona procenat tacno predvidjenih je 70.44
#%% Obuka
classifier1 = MLPClassifier(hidden_layer_sizes=(50,50,50), activation='tanh',
                              solver='adam', batch_size=50, learning_rate='constant', 
                              learning_rate_init=0.001, max_iter=50, shuffle=True,
                              random_state=42, early_stopping=True, n_iter_no_change=10,
                              validation_fraction=0.1, verbose=False)
classifier1.fit(X.values,y)
y_pred = classifier1.predict(data_test.iloc[:,:-1].values)
conf_mat = confusion_matrix(data_test.iloc[:,-1], y_pred, labels=labels_y)

#print(conf_mat)
disp = ConfusionMatrixDisplay(confusion_matrix =conf_mat,  display_labels=classifier.classes_)
disp.plot(cmap="BuGn", values_format='', xticks_rotation=90)  
plt.show()

print('procenat pogodjenih uzoraka: ', accuracy_score(data_test.iloc[:,-1], y_pred))
print('preciznost mikro: ', precision_score(data_test.iloc[:,-1], y_pred, average='micro'))
print('preciznost makro: ', precision_score(data_test.iloc[:,-1], y_pred, average='macro'))
print('osetljivost mikro: ', recall_score(data_test.iloc[:,-1], y_pred, average='micro'))
print('osetljivost makro: ', recall_score(data_test.iloc[:,-1], y_pred, average='macro'))
print('f mera mikro: ', f1_score(data_test.iloc[:,-1], y_pred, average='micro'))
print('f mera makro: ', f1_score(data_test.iloc[:,-1], y_pred, average='macro'))
#%%

#%% Model persistance
from sklearn import svm
from sklearn import datasets
#%%
clf = svm.SVC()
X, y= datasets.load_iris(return_X_y=True)
clf.fit(X, y)
SVC()
#%%
import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(X[0:1])
print(y[0])
#%%
 from joblib import dump, load
 dump(clf, 'svm.joblib') 
 #%%
dlf = load('svm.joblib') 