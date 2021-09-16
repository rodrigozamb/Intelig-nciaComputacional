import random
import numpy as np
import matplotlib.pyplot as plt


class TravelingSalesman:

    population = []

    cities = []

    def __init__(self,qntCidades,QntPopulacao,epochs):
        
        self.numberOfCities = qntCidades
        self.populationSize = QntPopulacao
        self.epochs = epochs

        self.createCities()

        for i in range(QntPopulacao):
            self.population.append( self.createElement() )

    def createCities(self):
        self.cities = [[random.randint(0,11),random.randint(0,11)] for i in range(self.numberOfCities)]

    def createElement(self):
        return [[random.randint(0,11),random.randint(0,11)] for i in range(self.numberOfCities)]

    def plotCities(self,cities):
        print(cities)
        for i in range(self.numberOfCities):
            plt.annotate("Cidade "+str(i+1),(cities[i][0],cities[i][1]))
        cidadesX = [cidade[0] for cidade in cities]
        cidadesY = [cidade[1] for cidade in cities]
        plt.plot(cidadesX,cidadesY)
        plt.show()

    def calculateDistance(self,city1,city2):
        return np.sqrt( (city2[0]-city1[0])**2 + (city2[1]-city1[1])**2 )

    def evaluateElement(self,element):
        value = 0.0
        for i in range(len(element)-1):
            value += self.calculateDistance(element[i],element[i+1])
        print(f'Element Value = {value}')
        return value


ts = TravelingSalesman(5,5,10)
ts.plotCities(ts.cities)

# for c in ts.population:
#     ts.evaluateElement(c)