import math
import pandas

from typing import Union, Optional

from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler


def mybank():
    bank = pandas.read_csv('../dados/bank/bank.csv', sep=';')

    colunas_numericas = ['age', 'balance', 'day', 'duration',
                         'campaign', 'pdays', 'previous']

    bank_classe = bank['y']
    bank_numericos = bank[colunas_numericas]
    bank_categoricos = bank.drop(columns=colunas_numericas + ['y'])

    normalizer = MinMaxScaler()
    bank_numericos_normal = normalizer.fit_transform(bank_numericos)
    bank_categoricos_normal = pandas.get_dummies(bank_categoricos)

    bank_normal = pandas.DataFrame(bank_numericos_normal,
                                   columns=colunas_numericas)
    bank_normal = bank_normal.join(bank_categoricos_normal)

    # balanceamento

    print('Bank antes de balencear:', Counter(bank['y']))
    # Counter({'no': 4000, 'yes': 521})
    # Então sim, precisa!
    oversample = SMOTE()
    x, y = oversample.fit_resample(bank_normal, bank_classe)
    print('Bank após balancear', Counter(y))

    # inferência

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    # bank_rf = RandomForestClassifier().fit(x_train, y_train)
    # y_previsto = bank_rf.predict(x_test)

    # bank_rf = RandomForestClassifier().fit(x_train, y_train)
    # y_previsto = bank_rf.predict(x_test)

    # Acurácia do modelo
    print('Acurácia do bank:', metrics.accuracy_score(y_test, y_previsto))





    # Usa o bank full como novas instâncias para o modelo
    # mas normaliza ele primeiro
    bank_full = pandas.read_csv('../dados/bank/bank-full.csv', sep=';')

    bank_full_classe = bank_full['y']
    bank_full_numericos = bank_full[colunas_numericas]
    bank_full_categoricos = bank_full.drop(columns=colunas_numericas + ['y'])

    bank_full_numericos_normal = normalizer.fit_transform(bank_full_numericos)
    bank_full_categoricos_normal = pandas.get_dummies(bank_full_categoricos)
    bank_full_normal = pandas.DataFrame(bank_full_numericos_normal,
                                        columns=colunas_numericas)
    bank_full_normal = bank_full_normal.join(bank_full_categoricos_normal)

    # Balancear Bank Full

    print('Bank full antes de balancear:', Counter(bank_full_classe))
    x, y = oversample.fit_resample(bank_full_normal, bank_full_classe)
    print('Bank full após balancear:', Counter(y))

    # faz a inferência do bank-full
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    bank_rf = RandomForestClassifier().fit(x_train, y_train)
    y_previsto = bank_rf.predict(x_test)

    # Acurácia do modelo usando o full
    print('Acurácia do bank full', metrics.accuracy_score(y_test, y_previsto))


def mysmote():
    # aqui não lê o output, só os dados (X)
    # fert = pd.read_csv('fertility.csv', usecols=[x for x in range(0,9)])

    fert = pandas.read_csv('../dados/fertility_diagnosis.txt')
    fert.atributos = fert.drop(columns=['Output'])
    fert.classes = fert['Output']
    # data = fert.atributos.to_numpy()
    classes_count = collections.Counter(fert.classes)
    oversample = SMOTE()
    x, y = oversample.fit_resample(fert.atributos, fert.classes)
    print(x)
    print(y)
    print(classes_count)
    print(collections.Counter(y))

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25)
    rf_fertility = RandomForestClassifier().fit(X_train, Y_train)
    y_previsto = rf_fertility.predict(X_test)
    print(y_previsto)
    print(metrics.accuracy_score(Y_test, y_previsto))


class Attributes:

    def __init__(self, **attributes: Union[int, float]) -> None:
        for attr, value in attributes.items():
            self.attr = value

    def __iter__(self):
        return self.__dict__.__iter__()

    def __str__(self):
        clsname = f'Classe {self.__class__.__name__}'
        clsattr = self.__dict__
        return f'{clsname}: {clsattr}'

    def __setattr__(self, name: str, value: Union[int, float]) -> None:
        if not isinstance(name, str):
            raise TypeError('O nome do atributo deve ser str')

        if not (isinstance(value, int) or isinstance(value, float)):
            raise TypeError('O valor do atributo deve ser int ou float')

        self.__dict__[name] = value

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def copy(self, negate: Optional[bool] = False):
        new = type(self)()

        if negate:
            new.__dict__.update({name: -value for name, value in self.items()})
        else:
            new.__dict__.update(self.__dict__)

        return new


class Instance(Attributes):

    def __init__(self, **attributes):
        super().__init__(**attributes)


class Logit:

    def __init__(self, alpha: Union[int, float],
                 coefficients: Attributes) -> None:
        self.alpha = alpha
        self.coefficients = coefficients

    def bvalue(self, **bvalues: dict) -> None:
        for b, value in bvalues.items():
            self.bvalues[b] = value

    def logit(self, instance: Instance) -> float:
        sumprod = sum([a*b for a, b in zip(xvalues.values(), self.bvalues.values())])
        return sumprod
        #return 1 / (1 + math.e**(-self.alpha + sumprod))


class Class(Logit):

    def __init__(self, name, alpha, **attributes):
        super(Instance, self).__init__(**attributes)
        super(Logit, self).__init__(alpha=alpha, **attributes)
        self.name = name

    def __str__(self):
        clsname = f'{self.__module__.__qualname__} <{self.name}>'
        clsattr = self.attributes
        return f'{clsname}: {clsattr}'

    def instance(self, instance):
        if instance.keys() != self.attributes.keys():
            raise ValueError('An istance must have the same attributes as the'
                             'Class')
        return self.logit(**instance)


class Model:

    def __init__(self, name, class1, class2):
        self.name = name
        self.class1 = class1
        self.class2 = class2


if __name__ == '__main__':

    a = Attributes()
    a.teste = -99.9
    a.bla = 2.22
    print(a)
    print(a.copy(negate=True))
    raise SystemExit

    # mybank()
    # raise SystemExit

    # mysmote()
    # raise SystemExit

    # c1 = Class(name='tested_negative', alpha=4.18)
    # c1.attribute(preg=-0.06)
    # c1.attribute(plas=-0.02)
    # c1.attribute(pres=0.01)
    # c1.attribute(insu=0)
    # c1.attribute(mass=-0.04)
    # c1.attribute(pedi=-0.47)
    # c1.attribute(age=-0.01)
    #
    # c2 = Class(name='tested_positive', alpha=-4.18)
    # c2.attribute(preg=0.06)
    # c2.attribute(plas=0.02)
    # c2.attribute(pres=-0.01)
    # c2.attribute(insu=-0)
    # c2.attribute(mass=0.04)
    # c2.attribute(pedi=0.47)
    # c2.attribute(age=0.01)

    c1 = Class(name='N', alpha=0.88)
    c2 = Class(name='O', alpha=-0.88)

    c1.attribute(season=-0.33,
                 age=-3.34,
                 childish_diseases=-0.14,
                 accident=0.78,
                 surgical_intervention=0.33,
                 high_fevers=0.48,
                 alcohol_consumption=1.87,
                 smoking=-0.1,
                 hours_sitting=-1.12)

    # c2.attribute(season=0.33,
    #              age=3.34,
    #              childish_diseases=0.14,
    #              accident=-0.78,
    #              surgical_intervention=-0.33,
    #              high_fevers=-0.48,
    #              alcohol_consumption=-1.87,
    #              smoking=0.1,
    #              hours_sitting=1.12)

    r1 = c1.instance(season=-0.33,
                     age=0.69,
                     childish_diseases=0,
                     accident=1,
                     surgical_intervention=1,
                     high_fevers=0,
                     alcohol_consumption=0.8,
                     smoking=0,
                     hours_sitting=0.88)

    # r2 = c2.instance(season=-0.33,
    #             age=0.69,
    #             childish_diseases=0,
    #             accident=1,
    #             surgical_intervention=1,
    #             high_fevers=0,
    #             alcohol_consumption=0.8,
    #             smoking=0,
    #             hours_sitting=0.88)

    print(r1)
    # m = Model('aula', c1, c2)

    # d = Diabetes(
    #     aplha=(),
    #     preg=(-0.06, -0.02),
    #     plas=(-0.02,  0.02),
    #     pres=( 0.01, -0.01),
    #     insu=( 0,    -0),
    #     mass=(-0.04,  0.04),
    #     pedi=(-0.47,  0.47),
    #      age=(-0.01, -0.01)
    # )
