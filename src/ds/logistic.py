import numpy
import math
import pandas


from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


class MyLogistcRegresion:
    """ Notação:

    x = variável de entrada
    y = variável de saída
    h(x) = hipótese ou função que converte ou calcula uma saída y baseada na entrada x

    (x, y) = um elemento do conjunto de treinamento
    m = quantidade de elementros de treinamento

    (xⁱ, yⁱ) = o elemento da posição "i" do conjunto de elementos de treinamento
    """

    def __init__(self) -> None:
        self._theta = None
        self._cost = None

    """ Prediz se x será 1 ou 0 utilizando a fórmula logística:
        
                     1      
        f(x) = ─────────────
                    (-θ ⋅ x)
               1 + e        

    O parâmetro θ (theta) deve ser previamente calculado com o método fit()
    """
    def predict(self, x: numpy.array) -> numpy.array:
        pass

    """
                  m                    
                 ___                   
            1    ╲        
        θ = ─ ⋅  ╱    cost(xⁱ, yⁱ)
            m    ‾‾‾                   
                i = 0                  

        m = quantidade de elementos x  
                                       
        cost(xⁱ, yⁱ) = -log(x)       se y = 1    
        cost(xⁱ, yⁱ) = -log(1 - x)   se y = 0
    """
    @property
    def cost(self) -> numpy.array:
        if not self._cost:
            c = []
            for x, y in zip(self._x, self._y):
                if y == 1:
                    c.append(-math.log(x))
                if y == 0:
                    c.append(-math.log(1 - x))
                self._cost = numpy.array(c)
        return numpy.array(c)

    @property
    def theta(self):
        if not self._theta:
            m = len(self._x)

            self._theta = (1 / m) * self.cost.sum()

        return self._theta

    def fit(self, x: numpy.array, y: numpy.array):
        self._x = x
        self._y = y


if __name__ == '__main__':
    sklog = LogisticRegression()
    mylog = MyLogistcRegresion()

    normalizer = MinMaxScaler()
    oversample = SMOTE()

    data = pandas.read_csv('../../dados/bank/bank.csv', sep=';')[['age', 'y']]
    data.age = normalizer.fit_transform(data[['age']])

    x, y = oversample.fit_resample(data[['age']], data[['y']])
    x = x.age.to_numpy().reshape(-1, 1)
    y = y.y.to_numpy().flatten()

    sklog.fit(x, y)
    test = numpy.array([[0.1], [0.2], [0.3], [0.4], [0.5],
                        [0.6], [0.7], [0.8], [0.9], [1.0]])
    print(sklog.predict(test))



