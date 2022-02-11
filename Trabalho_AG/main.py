import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import time
import readData
# fri26DistanceMatrix = [[0, 83, 93, 129, 133, 139, 151, 169, 135, 114, 110, 98, 99, 95, 81, 152, 159, 181, 172, 185, 147, 157, 185, 220, 127, 181], [83, 0, 40, 53, 62, 64, 91, 116, 93, 84, 95, 98, 89, 68, 67, 127, 156, 175, 152, 165, 160, 180, 223, 268, 179, 197], [93, 40, 0, 42, 42, 49, 59, 81, 54, 44, 58, 64, 54, 31, 36, 86, 117, 135, 112, 125, 124, 147, 193, 241, 157, 161], [129, 53, 42, 0, 11, 11, 46, 72, 65, 70, 88, 100, 89, 66, 76, 102, 142, 156, 127, 139, 155, 180, 228, 278, 197, 190], [133, 62, 42, 11, 0, 9, 35, 61, 55, 62, 82, 95, 84, 62, 74, 93, 133, 146, 117, 128, 148, 173, 222, 272, 194, 182], [139, 64, 49, 11, 9, 0, 39, 65, 63, 71, 90, 103, 92, 71, 82, 100, 141, 153, 124, 135, 156, 181, 230, 280, 202, 190], [151, 91, 59, 46, 35, 39, 0, 26, 34, 52, 71, 88, 77, 63, 78, 66, 110, 119, 88, 98, 130, 156, 206, 257, 188, 160], [169, 116, 81, 72, 61, 65, 26, 0, 37, 59, 75, 92, 83, 76, 91, 54, 98, 103, 70, 78, 122, 148, 198, 250, 188, 148], [135, 93, 54, 65, 55, 63, 34, 37, 0, 22, 39, 56, 47, 40, 55, 37, 78, 91, 62, 74, 96, 122, 172, 223, 155, 128], [114, 84, 44, 70, 62, 71, 52, 59, 22, 0, 20, 36, 26, 20, 34, 43, 74, 91, 68, 82, 86, 111, 160, 210, 136, 121], [110, 95, 58, 88, 82, 90, 71, 75, 39, 20, 0, 18, 11, 27, 32, 42, 61, 80, 64, 77, 68, 92, 140, 190, 116, 103], [98, 98, 64, 100, 95, 103, 88, 92, 56, 36, 18, 0, 11, 34, 31, 56, 63, 85, 75, 87, 62, 83, 129, 178, 100, 99], [99, 89, 54, 89, 84, 92, 77, 83, 47, 26, 11, 11, 0, 23, 24, 53, 68, 89, 74, 87, 71, 93, 140, 189, 111, 107], [95, 68, 31, 66, 62, 71, 63, 76, 40, 20, 27, 34, 23, 0, 15, 62, 87, 106, 87, 100, 93, 116, 163, 212, 132, 130], [81, 67, 36, 76, 74, 82, 78, 91, 55, 34, 32, 31, 24, 15, 0, 73, 92, 112, 96, 109, 93, 113, 158, 205, 122, 130], [152, 127, 86, 102, 93, 100, 66, 54, 37, 43, 42, 56, 53, 62, 73, 0, 44, 54, 26, 39, 68, 94, 144, 196, 139, 95], [159, 156, 117, 142, 133, 141, 110, 98, 78, 74, 61, 63, 68, 87, 92, 44, 0, 22, 34, 38, 30, 53, 102, 154, 109, 51], [181, 175, 135, 156, 146, 153, 119, 103, 91, 91, 80, 85, 89, 106, 112, 54, 22, 0, 33, 29, 46, 64, 107, 157, 125, 51], [172, 152, 112, 127, 117, 124, 88, 70, 62, 68, 64, 75, 74, 87, 96, 26, 34, 33, 0, 13, 63, 87, 135, 186, 141, 81], [185, 165, 125, 139, 128, 135, 98, 78, 74, 82, 77, 87, 87, 100, 109, 39, 38, 29, 13, 0, 68, 90, 136, 186, 148, 79], [147, 160, 124, 155, 148, 156, 130, 122, 96, 86, 68, 62, 71, 93, 93, 68, 30, 46, 63, 68, 0, 26, 77, 128, 80, 37], [157, 180, 147, 180, 173, 181, 156, 148, 122, 111, 92, 83, 93, 116, 113, 94, 53, 64, 87, 90, 26, 0, 50, 102, 65, 27], [185, 223, 193, 228, 222, 230, 206, 198, 172, 160, 140, 129, 140, 163, 158, 144, 102, 107, 135, 136, 77, 50, 0, 51, 64, 58], [220, 268, 241, 278, 272, 280, 257, 250, 223, 210, 190, 178, 189, 212, 205, 196, 154, 157, 186, 186, 128, 102, 51, 0, 93, 107], [127, 179, 157, 197, 194, 202, 188, 188, 155, 136, 116, 100, 111, 132, 122, 139, 109, 125, 141, 148, 80, 65, 64, 93, 0, 90], [181, 197, 161, 190, 182, 190, 160, 148, 128, 121, 103, 99, 107, 130, 130, 95, 51, 51, 81, 79, 37, 27, 58, 107, 90, 0]]

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

        # self.createCities()

        for i in range(QntPopulacao):  # pra cada individuo da população
            self.population.append(self.createElement())  # população inicial

        # self.createDistanceMatrix()
        
        # self.distanceMatrix = fri26DistanceMatrix

    # Função que cria a matriz de distancia das cidades

    def createDistanceMatrix(self):

        for c1 in self.cities:
            aux = []
            for c2 in self.cities:
                aux.append(self.calculateDistance(c1, c2))
            self.distanceMatrix.append(aux)

    def createCities(self):
        # self.cities = [[random.randint(0,10), random.randint(0,10)] for i in range(self.numberOfCities)]
        for i in range(self.numberOfCities):
            city = [random.randint(0, 10), random.randint(0, 10)]
            while (city in self.cities):
                city = [random.randint(0, 10), random.randint(0, 10)]
            self.cities.append(city)

    # Função que cria um elemento de forma aleatória
    def createElement(self):
        el = []
        while len(el) < self.numberOfCities:
            city = random.randint(0, self.numberOfCities-1)
            if city not in el:
                el.append(city)
        return el

    # Função que plota as cidades e o caminho feito
    def plotCities(self, cities):
        print(cities)
        for i in cities:
            plt.annotate("Cidade " + str(i),
                         (self.cities[i][0], self.cities[i][1]))
        cidadesX = [self.cities[i][0] for i in cities]
        cidadesY = [self.cities[i][1] for i in cities]
        cidadesX.append(cidadesX[0])
        cidadesY.append(cidadesY[0])
        plt.plot(cidadesX, cidadesY)
        plt.show()

    def calculateDistance(self, city1, city2):
        return np.sqrt((city2[0]-city1[0])**2 + (city2[1]-city1[1])**2)

    # Função de avaliação de elemento, utilizada pra selecionar os melhores elementos da população
    def evaluateElement(self, element):
        value = 0.0
        for i in range(len(element)-1):
            value += self.distanceMatrix[element[i]][element[i+1]]
            # value += self.calculateDistance(self.cities[element[i]],self.cities[element[i+1]])
        value += self.distanceMatrix[element[len(element)-1]][element[0]]
        return value

    # Função teste de cruzamento, (na há cruzamento, não consegui fazer)
    def normalCross(self, el1, el2):

        return self.createElement(), self.createElement()

    def pmx_cx(self, p1, p2):

        a = random.randint(0, self.numberOfCities-2)
        b = random.randint(a+1, self.numberOfCities-1)

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

        for i in range(a, b+1):
            f1[i], f2[i] = f2[i], f1[i]

        irange = list(range(0, a)) + list(range(b+1, numCities))

        for i in irange:
            x = f1[i]
            while idx2[x] >= a and idx2[x] <= b:
                x = f2[idx2[x]]
            f1[i] = x

            x = f2[i]
            while idx1[x] >= a and idx1[x] <= b:
                x = f1[idx1[x]]
            f2[i] = x

        return f1, f2

    def order_cx(self, p1, p2):

        a = random.randint(0, self.numberOfCities-2)
        b = random.randint(a+1, self.numberOfCities-1)

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

    def indexOf(self, arr, x):
        for a in range(0, arr.__len__()):
            if arr[a] == x:
                return a
        return -1

    def findUnusedIndexValues(self, p, f):
        res = list()
        for a in p:
            if self.indexOf(f, a) == -1:
                res.append(a)
        return res

    def cycle2_cx(self, p1, p2):

        p1_cpy = [None] * p1.__len__()
        p2_cpy = [None] * p2.__len__()

        for i in range(len(p1)):
            p1_cpy[i] = p1[i]
        for i in range(len(p2)):
            p2_cpy[i] = p2[i]

        print('hellol shakoob')
        f1 = [None] * p1_cpy.__len__()
        f2 = [None] * p2_cpy.__len__()
        i1 = 0
        i2 = 0
        initial = p1_cpy[0]
        f1[i1] = p2_cpy[0]
        i1 += 1
        # latestUpdated2 = p2_cpy[0]
        check = 1

        while i1 < p1_cpy.__len__() and i2 < p2_cpy.__len__():
            index1 = self.indexOf(p1_cpy, f1[i1-1])
            index1 = self.indexOf(p1_cpy, p2_cpy[index1])
            latestUpdated2 = p2_cpy[index1]
            if latestUpdated2 == initial:
                f2[i2] = latestUpdated2
                i2 += 1
                # print("cycle detected")
                check = 0
                res1 = self.findUnusedIndexValues(p1_cpy, f1)
                res2 = self.findUnusedIndexValues(p2_cpy, f2)
                # print(res1,res2)
                ans1, ans2 = self.cycle2_cx(res1, res2)
                f1[i1:] = ans1
                f2[i2:] = ans2
                check = 0
                break
            else:
                f2[i2] = p2_cpy[index1]
                i2 += 1
                index1 = self.indexOf(p1_cpy, f2[i2-1])
                f1[i1] = p2_cpy[index1]
                i1 += 1
        if check:
            index1 = self.indexOf(p1_cpy, f1[i1 - 1])
            index1 = self.indexOf(p1_cpy, p2_cpy[index1])
            latestUpdated2 = p2_cpy[index1]
            f2[i2] = latestUpdated2
            i2 += 1
        return f1, f2

    def CX2(self, p1, p2):
        o1, o2 = [-1]*self.numberOfCities, [-1]*self.numberOfCities

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
                o2[i] = p2[p1.index(p2[p1.index(o1[i])])]
                # print(o1,o2)
        return o1, o2

    def handle_conversion(self):
        occ = self.population.count(self.population[0])
        if (occ/self.populationSize >= 0.90):
            return True
        else:
            return False

    def calculateMedia(self):
        s = 0.0
        for i in self.population:
            s += self.evaluateElement(i)
        return s/self.populationSize

    def roullete(self):
        p1, p2 = random.randint(
            0, self.populationSize-1), random.randint(0, self.populationSize-1)
        while(p1 == p2):
            p1, p2 = random.randint(
                0, self.populationSize-1), random.randint(0, self.populationSize-1)
        return self.population[p1], self.population[p2]

    def mutate(self, elem):

        x = random.uniform(0, 1)

        if(x <= self.mutationProb):
            a = random.randint(0, self.numberOfCities-1)
            b = random.randint(0, self.numberOfCities-1)

            while(a == b):
                b = random.randint(0, self.numberOfCities-1)

            aux = elem[a]
            elem[a] = elem[b]
            elem[b] = aux

    def neighbor_mutate(self, elem):
        x = random.uniform(0, 1)

        if(x <= self.mutationProb):
            a = random.randint(0, self.numberOfCities-2)
            aux = elem[a]
            elem[a] = elem[a+1]
            elem[a+1] = aux

    # Simula a evolução , ou seja, o passar das épocas com a evolução da população
    def simulate(self):

        for epoch in range(self.epochs):
            print(f'Epoch {epoch}')
            # print("População inicial da epoca")
            # print(self.population)
            filhos = []

            for j in range(int((self.crossoverProb*self.populationSize)/2)):
                p1, p2 = self.roullete()
                # print("Pais")
                # print(p1,p2)
                # escolher os pais para cruzamento (roleta?)
                f1, f2 = self.CX2(p1, p2)
                # f1,f2 = self.normalCross(p1, p2) # escolher os pais para cruzamento (roleta?)
                # print(p1,p2)

                self.mutate(f1)
                self.mutate(f2)

                filhos.append(f1)
                filhos.append(f2)
                
            # print('chega aqui')

            for f in filhos:
                self.population.append(f)
            # print("População junto com filhos")
            # print(self.population)
            # for i in range(len(self.population)):
            #     print(f'pop[{i}] = {self.evaluateElement(self.population[i])}')

            self.population.sort(key=self.evaluateElement)
            # print(self.population)

            while(len(self.population) > self.populationSize):
                self.population.pop()

            # for i in range(len(self.population)):
            #     print(f'pop[{i}] = {self.evaluateElement(self.population[i])}')
            # if(self.handle_conversion()):
            #     break

def loadAssym(filename='../data/assymetric/kro124p', length=100):
    distanceMatrix = readData.readAsymmetric(filename, length)

    print(distanceMatrix)

    # for i in range(len(distanceMatrix)):
    #     aux = []
    #     for j in range(len(distanceMatrix[i])):
    #         aux.append([i,j])
    #     distanceMatrix[i] = aux

    # print(distanceMatrix)

    return distanceMatrix

# Função principal do programa
def main():
    atsp_name = [
        # "ftv33",
        # "ft53",
        # "ftv38",
        "ftv170",
        # "kro124p",
        # "rbg323","rbg358","rbg403","rbg443"
    ]
    lengths = [
        # 34,
        # 53,
        # 39,
        171,
        # 100,
        # 323, 358, 403, 443
    ]

    for i in range(len(atsp_name)):
        f = open("results_cycle2/"+atsp_name[i]+'.txt', "a")
        f.write(f'---------{atsp_name[i]} : {lengths[i]}---------\n\n')
        f.close()

        cost = 0
        best = 10000000
        worst = 0
        mean_time = 0

        for j in range(10):
            ts = TravelingSalesman(lengths[i], 200, 1000, 0.1, 0.8)
            ts.distanceMatrix = loadAssym('../data/assymetric/'+atsp_name[i], lengths[i])
            # ts.distanceMatrix = fri26DistanceMatrix
            start_time = time.time()
            ts.simulate()
            cost = ts.evaluateElement(ts.population[0])
            final_time = time.time()

            total_time = final_time - start_time
            mean_time += total_time

            if(cost < best):
                best = cost
            elif(cost > worst):
                worst = cost

            f = open("results_cycle2/"+atsp_name[i]+'.txt', "a")
            f.write(f'Execução {j+1}--------------------------------\n\n')
            f.write(f'Melhor custo : {best}\n\n')
            f.write(f'Pior custo : {worst}\n\n')
            f.write(f'Tempo : {total_time}\n\n')
            f.close()
    

        f = open("results_cycle2/"+atsp_name[i]+'.txt', "a")
        f.write(f'----------------------------------------------\n')
        f.write(f'Melhor custo : {best}\n\n')
        f.write(f'Pior custo : {worst}\n\n')
        f.write(f'Media custo : {ts.calculateMedia()}\n\n')
        f.write(f'Tempo médio : {mean_time/10}\n\n')
        f.close()

main()
