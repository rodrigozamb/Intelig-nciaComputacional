import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

class TravelingSalesman:

    population = []
    distanceMatrix = []
    cities = []

    def __init__(self, qntCidades, QntPopulacao, epochs, mutprob, crossprob):
        
        self.numberOfCities = qntCidades
        self.populationSize = QntPopulacao
        self.epochs = epochs
        self.mutationProb = mutprob
        self.crossoverProb = crossprob

        self.createCities()
        
        for i in range(QntPopulacao): # pra cada individuo da população
            self.population.append( self.createElement() ) # população inicial

        self.createDistanceMatrix()


    # Função que cria a matriz de distancia das cidades
    def createDistanceMatrix(self):

        for c1 in self.cities:
            aux = []
            for c2 in self.cities:
                aux.append( self.calculateDistance(c1,c2) )
            self.distanceMatrix.append(aux)

    def createCities(self):
        self.cities = [[random.randint(0,10), random.randint(0,10)] for i in range(self.numberOfCities)]

    # Função que cria um elemento de forma aleatória
    def createElement(self):
        el = []
        while len(el) < self.numberOfCities:
            i = random.randint(0, self.numberOfCities-1)
            if i not in el:
                el.append(i)
        return el

    # Função que plota as cidades e o caminho feito
    def plotCities(self,cities):
        print(cities)
        for i in cities:
            plt.annotate("Cidade " + str(i), (self.cities[i][0], self.cities[i][1]))
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

    def pmx_cx(self, p1, p2, a, b):

        f1, f2 = [0]*self.numberOfCities, [0]*self.numberOfCities

        for i in range(len(p1)):
            f1[i] = p1[i]
        for i in range(len(p2)):
            f2[i] = p2[i]

        numCities = len(f1)
        idx1 = [0] * numCities
        idx2 = [0] * numCities

        for i, x in enumerate(f1):
            idx1[x] = i
        for i, x in enumerate(f2):
            idx2[x] = i

        for i in range(a, b+1) :
            f1[i], f2[i] = f2[i], f1[i]

        irange = list(range(0,a)) + list(range(b+1, numCities))

        for i in irange:
            x = f1[i]
            while idx2[x] >=a and idx2[x] <= b :
                x = f2[idx2[x]]
            f1[i] = x

            x = f2[i]
            while idx1[x] >= a and idx1[x] <= b:
                x = f1[idx1[x]]
            f2[i] = x

        return f1, f2

    # FUnção principal do programa
    def main(self):
        self.simulate()
        self.plotCities(self.population[0])

    def roullete(self):
        p1,p2 = random.randint(0,self.populationSize-1),random.randint(0,self.populationSize-1)
        while(p1 == p2):
            p1,p2 = random.randint(0,self.populationSize-1),random.randint(0,self.populationSize-1)
        return self.population[p1],self.population[p2]
        
    def mutate(self,elem):

        x = random.uniform(0, 1)
            
        if(x<=self.mutationProb):
            a = random.randint(0,self.numberOfCities-1)
            b = random.randint(0,self.numberOfCities-1)

            while(a==b):
                b = random.randint(0,self.numberOfCities-1)
                
            aux = elem[a]
            elem[a] = elem[b]
            elem[b] = aux


    # Simula a evolução , ou seja, o passar das épocas com a evolução da população
    def simulate(self):

        for epoch  in range(self.epochs):
            print(f'Epoch {epoch}')
            # print("População inicial da epoca")
            # print(self.population)
            filhos = []
            
            for j in range(int((self.crossoverProb*self.populationSize)/2)):
                p1,p2 = self.roullete()
                # print("Pais")
                # print(p1,p2)
                f1,f2 = self.pmx_cx(p1, p2, 3, 5) # escolher os pais para cruzamento (roleta?)
                # f1,f2 = self.normalCross(p1, p2) # escolher os pais para cruzamento (roleta?)
                # print(p1,p2)
                
                self.mutate(f1)
                self.mutate(f2)

                filhos.append(f1)
                filhos.append(f2)
            
            for f in filhos:
                self.population.append(f)
            # print("População junto com filhos")
            print(self.population)
            for i in range(len(self.population)):
                print(f'pop[{i}] = {self.evaluateElement(self.population[i])}')
            
            self.population.sort(key=self.evaluateElement)
            print(self.population)

            while(len(self.population)>self.populationSize):
                self.population.pop()
            
            for i in range(len(self.population)):
                print(f'pop[{i}] = {self.evaluateElement(self.population[i])}')
            # if epoch == 1:
            #     break

# params: qntCidades, QntPopulacao, epochs, mutprob, crossprob
ts = TravelingSalesman(8, 30, 10, 0.1, 0.8)
# print(ts.population)
# for i in range(4):
#     p1,p2 = ts.roullete()
#     print("pai 1 = ", p1)
#     print("pai 2 = ", p2)
    

ts.main()

# print('------testing cross over------')
# s1 = ts.population[0]
# s2 = ts.population[1]
# print('s1, s2 before cross over')
# print(s1, s2)
# s1,s2 = ts.pmx_cx(s1, s2, 3, 5)
# print('s1, s2 after cross over')
# print(s1, s2)


### Arrumar a matriz de distancias a ser utilizada, usar proababilidade de cross, gerar os filhos para depois selecionar os melhores, criar os 4 tipos de cross, finalizar a simulação das epocas e da evolução