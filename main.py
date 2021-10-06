import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import readData
import cx2
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
        
        for i in range(QntPopulacao):
            self.population.append( self.createElement() )

        # self.createDistanceMatrix()
        self.distanceMatrix = readData.readAsymmetric('data/assymetric/kro124p', 100)

    def createDistanceMatrix(self):

        for c1 in self.cities:
            aux = []
            for c2 in self.cities:
                aux.append( self.calculateDistance(c1,c2) )
            self.distanceMatrix.append(aux)

    def createCities(self):
        for i in range(self.numberOfCities):
            city = [random.randint(0,10), random.randint(0,10)]
            while (city in self.cities):
                city = [random.randint(0,10), random.randint(0,10)]
            self.cities.append(city)

    # Função que cria uma cidade de forma aleatória
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
        cidadesX.append(cidadesX[0])
        cidadesY.append(cidadesY[0])
        plt.plot(cidadesX,cidadesY)
        plt.show()

    def calculateDistance(self,city1,city2):
        return np.sqrt( (city2[0]-city1[0])**2 + (city2[1]-city1[1])**2 )

    # Função de avaliação de elemento, utilizada pra selecionar os melhores elementos da população
    def evaluateElement(self,element):
        value = 0.0
        for i in range(len(element)-1):
            value += self.distanceMatrix[element[i]][element[i+1]]
        value += self.distanceMatrix[element[len(element)-1]][element[0]]
        return value

    #Função teste de cruzamento
    def normalCross(self,el1,el2):

        return self.createElement(),self.createElement()

    def pmx_cx(self, p1, p2):

        a = random.randint(0,self.numberOfCities-2)
        b = random.randint(a+1,self.numberOfCities-1)

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

    def order_cx(self, p1, p2):

        a = random.randint(0,self.numberOfCities-2)
        b = random.randint(a+1,self.numberOfCities-1)

        f1, f2 = [0]*self.numberOfCities, [0]*self.numberOfCities

        for i in range(len(p1)):
            f1[i] = p1[i]
        for i in range(len(p2)):
            f2[i] = p2[i]

        ind_size = len(f1)

        indx1 = [0] * ind_size
        indx2 = [0] * ind_size
        for i, x in enumerate(f1):
            indx1[x] = i
        for i, x in enumerate(f2):
            indx2[x] = i


        f2_cpy = f2.copy()
        f1_cpy = f1.copy()

        for k in range(a, b + 1):
            f2_cpy[indx2[f1[k]]] = -1
            f1_cpy[indx1[f2[k]]] = -1

        ptr1 = ptr2 = (b + 1) % ind_size

        for i in range(ind_size - (b - a + 1)):

            while f2_cpy[ptr2] == -1:
                ptr2 = (ptr2 + 1) % ind_size
            while f1_cpy[ptr1] == -1:
                ptr1 = (ptr1 + 1) % ind_size

            f1[(b + i + 1) % ind_size] = f2_cpy[ptr2]
            f2[(b + i + 1) % ind_size] = f1_cpy[ptr1]

            ptr1 = (ptr1 + 1) % ind_size
            ptr2 = (ptr2 + 1) % ind_size

        return f1, f2

    def cycle_cx(self, p1, p2):

        f1, f2 = [0]*self.numberOfCities, [0]*self.numberOfCities

        for i in range(len(p1)):
            f1[i] = p1[i]
        for i in range(len(p2)):
            f2[i] = p2[i]

        ind_size = len(f1)
        o1 = [-1]*ind_size
        o2 = [-1]*ind_size

        indx1 = [0] * ind_size
        indx2 = [0] * ind_size
        for i, x in enumerate(f1):
            indx1[x] = i
        for i, x in enumerate(f2):
            indx2[x] = i

        i = 0
        o1[i] = f1[i]
        i = indx1[f2[i]]
        while o1[i] == -1 :
            o1[i] = f1[i]
            i = indx1[f2[i]]
        for i in range(ind_size) :
            if o1[i] == -1:
                o1[i] = f2[i]

        i = 0
        o2[i] = f2[i]
        i = indx2[f1[i]]
        while o2[i] == -1:
            o2[i] = f2[i]
            i = indx2[f1[i]]
        for i in range(ind_size):
            if o2[i] == -1:
                o2[i] = f1[i]

        f1[:] = o1[:]
        f2[:] = o2[:]


        return f1, f2

    def CX2(self,p1,p2):
        o1,o2 = [-1]*self.numberOfCities,[-1]*self.numberOfCities

        # print(p1,p2)
        f = True
        for i in range(self.numberOfCities):
            if((-1 not in o1 and -1 not in o2)):
                break
            
            if(p1[0] in o2):
                f = False
                reset = 0
                for k in p2:
                    if k not in o1:
                        reset = k
                        break

                o1[i] = reset
                o2[i] = p2[p1.index(p2[p1.index(o1[i])])]
                # print(o1,o2)
                continue


            if i == 0:
                o1[0] = p2[0]
                o2[i] = p2[p1.index(p2[p1.index(o1[i])])]
                # print(o1,o2)
            else:
                o1[i] = p2[p1.index(o2[i-1])]
                o2[i] = p2[  p1.index(p2[p1.index(o1[i])])]
                # print(o1,o2)
        return o1,o2

    def handle_conversion(self):
        occ = self.population.count(self.population[0])
        if (occ/self.populationSize >= 0.90):
            return True
        else:
            return False

    # Função principal do programa
    def main(self, exec):
        f = open("kro124p_cx2.txt","a")
        f.write(f'Execução : ${exec}\n')
        self.simulate()
        # self.plotCities(self.population[0])
        f.write(f'Melhor = {self.evaluateElement(self.population[0])}\n')
        f.write(f'Pior = {self.evaluateElement(self.population[self.populationSize-1])}\n')
        f.write(f'Media = {self.calculateMedia()}\n')
        f.write("\n")
        f.close()
        
    def calculateMedia(self):
        s = 0.0
        for i in self.population:
            s+=self.evaluateElement(i)
        return s/self.populationSize

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

    def neighbor_mutate(self,elem):
        x = random.uniform(0, 1)
            
        if(x<=self.mutationProb):
            a = random.randint(0,self.numberOfCities-2)
            aux = elem[a]
            elem[a] = elem[a+1]
            elem[a+1] = aux

    # Simula a evolução , ou seja, o passar das épocas com a evolução da população
    def simulate(self):

        for epoch  in range(self.epochs):
            print(f'Epoch {epoch}')
            filhos = []
            
            for j in range(int((self.crossoverProb*self.populationSize)/2)):
                p1,p2 = self.roullete()
                f1,f2 = self.CX2(p1, p2)
                
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
            # print(self.population)

            while(len(self.population)>self.populationSize):
                self.population.pop()
            
            if(self.handle_conversion()):
                break

if __name__ == "__main__":
    # params: qntCidades, QntPopulacao, epochs, mutprob, crossprob
    ts = TravelingSalesman(100, 200, 1000, 0.1, 0.8)

    for i in range(30):
        ts.main(i+1)
