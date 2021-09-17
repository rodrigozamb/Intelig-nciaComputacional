import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

class TravelingSalesman:

    population = []
    distanceMatrix = []
    cities = []

    def __init__(self,qntCidades,QntPopulacao,epochs,mutprob,crossprob):
        
        self.numberOfCities = qntCidades
        self.populationSize = QntPopulacao
        self.epochs = epochs
        self.mutationProb = mutprob
        self.crossoverProb = crossprob

        self.createCities()
        print(self.cities)
        for i in range(QntPopulacao):
            self.population.append( self.createElement())
        self.createDistanceMatrix()

    # Função que cria a matriz de distancia das cidades
    def createDistanceMatrix(self):

        for c1 in self.cities:
            aux = []
            for c2 in self.cities:
                aux.append( self.calculateDistance(c1,c2) )
            self.distanceMatrix.append(aux)

    def createCities(self):
        self.cities = [[random.randint(0,10),random.randint(0,10)] for i in range(self.numberOfCities)]

    # Função que cria um elemento do forma aleatória
    def createElement(self):
        el = []
        while len(el) < self.numberOfCities:
            i = random.randint(0,self.numberOfCities-1)
            if i not in el:
                el.append(i)
        return el

    # Função que plota as cidades e o caminho feito
    def plotCities(self,cities):
        print(cities)
        for i in cities:
            plt.annotate("Cidade "+str(i),(self.cities[i][0],self.cities[i][1]))
        cidadesX = [self.cities[i][0] for i in cities]
        cidadesY = [self.cities[i][1] for i in cities]
        plt.plot(cidadesX,cidadesY)
        plt.show()

    def calculateDistance(self,city1,city2):
        return np.sqrt( (city2[0]-city1[0])**2 + (city2[1]-city1[1])**2 )

    # Função de avaliação de elemento, utilizada pra selecionar os melhores elementos da população
    def evaluateElement(self,element):
        value = 0.0
        for i in range(len(element)-1):
            value += self.distanceMatrix[element[i]][element[i+1]]
            # value += self.calculateDistance(self.cities[element[i]],self.cities[element[i+1]])
        return value

    #Função teste de cruzamento, (na há cruzamento, não consegui fazer)
    def normalCross(self,el1,el2):

        return self.createElement(),self.createElement()

    # FUnção principal do programa
    def main(self):
        self.simulate()
        self.plotCities(self.population[0])

    # Simula a evolução , ou seja, o passar das épocas com a evolução da população
    def simulate(self):

        for i  in range(self.epochs):
            print(f'Epoch {i}')
            filhos = []
            for j in range(int((self.crossoverProb*self.populationSize)/2)):
                f1,f2 = self.normalCross(1,2)
                filhos.append(f1)
                filhos.append(f2)
            for f in filhos:
                self.population.append(f)
            self.population.sort(key=ts.evaluateElement)
            self.population = self.population[:self.populationSize]



ts = TravelingSalesman(7,5,20,0.2,0.8)
ts.main()

### Arrumar a matriz de distancias a ser utilizada, usar proababilidade de cross, gerar os filhos para depois selecionar os melhores, criar os 4 tipos de cross, finalizar a simulação das epocas e da evolução