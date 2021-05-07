import pandas
import sklearn
import collections
import imblearn
import os
import pickle


def atividade(csvfile, csvsep, col_num, col_cls, dropcol=None):
    # 1 Normalizar os dados do arquivo Breast Cancer
    # 1.1 A coluna deg_malig é numérica, as demais são categóricas
    # 2. Obter modelo (RandomForest)
    # 3. Gerar relatório de acurácia (Classification Report)
    # 4. Salvar o modelo em disco

    #
    # Carrega os dados do CSV
    #
    print('Carregando', csvfile)
    data = pandas.read_csv(csvfile, sep=csvsep)

    if dropcol:
        data = data.drop(columns=dropcol)

    for col in data:
        colnan = data[col].isnull().sum()
        if colnan > 0:
            print('{} tem {} NaN!'.format(col, colnan))

    #
    # Separa os dados categóricos, númericos e classe da base original
    #
    data_class = data[col_cls]
    data_num = data[col_num]
    data_cat = data.drop(columns=col_num + [col_cls])

    #
    # Realiza a normalização dos dados categóricos e numéricos
    #
    normalizer = sklearn.preprocessing.MinMaxScaler()

    data_num = normalizer.fit_transform(data_num)
    data_cat = pandas.get_dummies(data_cat)

    data_normal = pandas.DataFrame(data_num, columns=col_num)
    data_normal = data_normal.join(data_cat)

    #
    # Faz o balanceamento da classe
    #
    print('Antes de balancear:', collections.Counter(data_class))

    oversample = imblearn.over_sampling.SMOTE()
    x, y = oversample.fit_resample(data_normal, data_class)

    print('Depois de balancear', collections.Counter(y))

    #
    # Realiza a inferência
    #
    x_train, x_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(x, y, test_size=0.3)

    model = sklearn.ensemble.RandomForestClassifier().fit(x_train, y_train)
    y_predict = model.predict(x_test)

    # Acurácia do modelo
    print('Acurácia do modelo:',
          sklearn.metrics.accuracy_score(y_test, y_predict))

    class_report = sklearn.metrics.classification_report(
        y_test, y_predict, target_names=model.classes_
    )
    print(class_report)

    pname = '{}_model.pkl'.format(os.path.splitext(csvfile)[0])
    with open(pname, 'w+b') as pfile:
        pickle.dump(model, pfile)


if __name__ == '__main__':
    # csv = '../dados/bank/bank-full.csv'
    # sep = ';'
    # c_num = ['age', 'balance', 'day', 'duration']
    # c_num += ['campaign', 'pdays', 'previous']
    # c_cls = 'y'
    #
    # atividade(csv, sep, c_num, c_cls)
    #
    csv = '../dados/breast-cancer.csv'
    sep = ','
    c_num = ['deg-malig']
    c_cls = 'Class'

    atividade(csv, sep, c_num, c_cls)

    # csv = '../dados/hypothyroid.csv'
    # sep = ';'
    # # c_num = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
    # c_num = ['age']
    # c_cls = 'Class'
    #
    # atividade(csv, sep, c_num, c_cls,
    #           dropcol=['TBG', 'sex', 'TSH', 'T3', 'TT4', 'T4U', 'FTI'])
