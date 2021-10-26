from operator import attrgetter
import random
import matplotlib.pyplot as plt
import math
import readData
import sys
import time
class Graph:

	def __init__(self, amount_vertices):
		self.edges = {} 
		self.vertices = set()
		self.pheromoneMap = {}
		self.amount_vertices = amount_vertices 


	def addEdge(self, src, dest, cost = 0):
		if (src, dest) not in self.edges:
			self.edges[(src, dest)] = cost
			self.pheromoneMap[(src, dest)] = 0.0
			self.vertices.add(src)
			self.vertices.add(dest)

	def showPheromoneMap(self):
		for i in self.pheromoneMap:
			print(i[0],i[1],self.pheromoneMap[(i[0],i[1])])

	def showGraph(self):
		print('Showing the graph:\n')
		for edge in self.edges:
			print('%d linked in %d with cost %d' % (edge[0], edge[1], self.edges[edge]))

	def getCostPath(self, path):
		
		total_cost = 0
		for i in range(self.amount_vertices - 1):
			total_cost += self.edges[(path[i], path[i+1])]

		total_cost += self.edges[(path[self.amount_vertices - 1], path[0])]
		return total_cost

	def getRandomPaths(self, max_size,initial_pos):

		random_paths = []
		list_vertices = list(self.vertices)

		list_vertices.remove(initial_pos)
		list_vertices.insert(0, initial_pos)

		for i in range(max_size):
			list_temp = list_vertices[1:]
			random.shuffle(list_temp)
			list_temp.insert(0, initial_pos)

			# possível de repetir elementos
			if self.valid_path(list_temp):
				random_paths.append(list_temp)

		return random_paths
	
	def valid_path(self,path):

		for i in range(len(path)-1):
			if(path[i],path[i+1]) not in self.edges:
				return False
		
		return True

	def loadfri06(self):
		fri26DistanceMatrix = [[0, 83, 93, 129, 133, 139, 151, 169, 135, 114, 110, 98, 99, 95, 81, 152, 159, 181, 172, 185, 147, 157, 185, 220, 127, 181], [83, 0, 40, 53, 62, 64, 91, 116, 93, 84, 95, 98, 89, 68, 67, 127, 156, 175, 152, 165, 160, 180, 223, 268, 179, 197], [93, 40, 0, 42, 42, 49, 59, 81, 54, 44, 58, 64, 54, 31, 36, 86, 117, 135, 112, 125, 124, 147, 193, 241, 157, 161], [129, 53, 42, 0, 11, 11, 46, 72, 65, 70, 88, 100, 89, 66, 76, 102, 142, 156, 127, 139, 155, 180, 228, 278, 197, 190], [133, 62, 42, 11, 0, 9, 35, 61, 55, 62, 82, 95, 84, 62, 74, 93, 133, 146, 117, 128, 148, 173, 222, 272, 194, 182], [139, 64, 49, 11, 9, 0, 39, 65, 63, 71, 90, 103, 92, 71, 82, 100, 141, 153, 124, 135, 156, 181, 230, 280, 202, 190], [151, 91, 59, 46, 35, 39, 0, 26, 34, 52, 71, 88, 77, 63, 78, 66, 110, 119, 88, 98, 130, 156, 206, 257, 188, 160], [169, 116, 81, 72, 61, 65, 26, 0, 37, 59, 75, 92, 83, 76, 91, 54, 98, 103, 70, 78, 122, 148, 198, 250, 188, 148], [135, 93, 54, 65, 55, 63, 34, 37, 0, 22, 39, 56, 47, 40, 55, 37, 78, 91, 62, 74, 96, 122, 172, 223, 155, 128], [114, 84, 44, 70, 62, 71, 52, 59, 22, 0, 20, 36, 26, 20, 34, 43, 74, 91, 68, 82, 86, 111, 160, 210, 136, 121], [110, 95, 58, 88, 82, 90, 71, 75, 39, 20, 0, 18, 11, 27, 32, 42, 61, 80, 64, 77, 68, 92, 140, 190, 116, 103], [98, 98, 64, 100, 95, 103, 88, 92, 56, 36, 18, 0, 11, 34, 31, 56, 63, 85, 75, 87, 62, 83, 129, 178, 100, 99], [99, 89, 54, 89, 84, 92, 77, 83, 47, 26, 11, 11, 0, 23, 24, 53, 68, 89, 74, 87, 71, 93, 140, 189, 111, 107], [95, 68, 31, 66, 62, 71, 63, 76, 40, 20, 27, 34, 23, 0, 15, 62, 87, 106, 87, 100, 93, 116, 163, 212, 132, 130], [81, 67, 36, 76, 74, 82, 78, 91, 55, 34, 32, 31, 24, 15, 0, 73, 92, 112, 96, 109, 93, 113, 158, 205, 122, 130], [152, 127, 86, 102, 93, 100, 66, 54, 37, 43, 42, 56, 53, 62, 73, 0, 44, 54, 26, 39, 68, 94, 144, 196, 139, 95], [159, 156, 117, 142, 133, 141, 110, 98, 78, 74, 61, 63, 68, 87, 92, 44, 0, 22, 34, 38, 30, 53, 102, 154, 109, 51], [181, 175, 135, 156, 146, 153, 119, 103, 91, 91, 80, 85, 89, 106, 112, 54, 22, 0, 33, 29, 46, 64, 107, 157, 125, 51], [172, 152, 112, 127, 117, 124, 88, 70, 62, 68, 64, 75, 74, 87, 96, 26, 34, 33, 0, 13, 63, 87, 135, 186, 141, 81], [185, 165, 125, 139, 128, 135, 98, 78, 74, 82, 77, 87, 87, 100, 109, 39, 38, 29, 13, 0, 68, 90, 136, 186, 148, 79], [147, 160, 124, 155, 148, 156, 130, 122, 96, 86, 68, 62, 71, 93, 93, 68, 30, 46, 63, 68, 0, 26, 77, 128, 80, 37], [157, 180, 147, 180, 173, 181, 156, 148, 122, 111, 92, 83, 93, 116, 113, 94, 53, 64, 87, 90, 26, 0, 50, 102, 65, 27], [185, 223, 193, 228, 222, 230, 206, 198, 172, 160, 140, 129, 140, 163, 158, 144, 102, 107, 135, 136, 77, 50, 0, 51, 64, 58], [220, 268, 241, 278, 272, 280, 257, 250, 223, 210, 190, 178, 189, 212, 205, 196, 154, 157, 186, 186, 128, 102, 51, 0, 93, 107], [127, 179, 157, 197, 194, 202, 188, 188, 155, 136, 116, 100, 111, 132, 122, 139, 109, 125, 141, 148, 80, 65, 64, 93, 0, 90], [181, 197, 161, 190, 182, 190, 160, 148, 128, 121, 103, 99, 107, 130, 130, 95, 51, 51, 81, 79, 37, 27, 58, 107, 90, 0]]
		self.edges = {} 
		self.vertices = set() 
		self.amount_vertices = len(fri26DistanceMatrix[0])

		for i in range(len(fri26DistanceMatrix)):
			for j in range(len(fri26DistanceMatrix[i])):
				self.addEdge(i,j,fri26DistanceMatrix[i][j])

	def generateComplete(self):
		for i in range(self.amount_vertices):
			for j in range(self.amount_vertices):
				if i != j:
					weight = random.randint(1, 10)
					self.addEdge(i, j, weight)

	def loadAssym(self, filename='../data/assymetric/kro124p', length=100):
		distanceMatrix = readData.readAsymmetric(filename, length)
		self.edges = {}
		self.vertices = set() 
		self.amount_vertices = len(distanceMatrix[0])

		for i in range(len(distanceMatrix)):
			for j in range(len(distanceMatrix[i])):
				self.addEdge(i,j,distanceMatrix[i][j])

class Ant:

	def __init__(self, initial_pos):

		self.initial_pos = initial_pos
		self.last_pos = initial_pos
		self.path = [initial_pos]
		self.pathCost = 0.0

	def setPath(self, path):
		self.path = path

	def getPath(self):
		return self.path

	def setPathCost(self, pathCost):
		self.path = pathCost

	def getPathCost(self):
		return self.pathCost

	def move(self,newPos):
		self.path.append(newPos)
		self.last_pos = newPos

class ACO:

	def __init__(self, graph, iterations, ants,evaporation,alfa,beta):
		self.graph = graph 
		self.iterations = iterations 
		self.ants = ants 
		self.ants = [] 
		self.gbest = []
		self.evaporation = evaporation
		self.alfa = alfa
		self.beta = beta

	def getGBest(self):
		return self.gbest

	def showsAnts(self):

		print('Formigas do ACO...\n')
		for ant in self.ants:
			print(f'Path = {ant.getPath()}  -  Cost = {ant.getPathCost()}')
		print('')

	def sortAux(self,elem):
		x,y,z = elem
		return z

	def calculate_next_move(self, ant):
		roulette_wheel = 0.0
		unvisited_nodes = [node for node in range(self.graph.amount_vertices) if node not in ant.path]
		heuristic_total = 0.0

		for unvisited_node in unvisited_nodes:
			heuristic_total += self.graph.edges[(ant.path[-1],unvisited_node)]
		for unvisited_node in unvisited_nodes:
			roulette_wheel += pow(self.graph.pheromoneMap[(ant.path[-1],unvisited_node)], self.alfa) * pow((heuristic_total / self.graph.edges[(ant.path[-1],unvisited_node)]), self.beta)
		random_value = random.uniform(0.0, roulette_wheel)
		wheel_position = 0.0
		for unvisited_node in unvisited_nodes:
			wheel_position += pow(self.graph.pheromoneMap[(ant.path[-1],unvisited_node)], self.alfa) * pow((heuristic_total / self.graph.edges[(ant.path[-1],unvisited_node)]), self.beta)
			if wheel_position >= random_value:
				return unvisited_node

	def evaporePheromoneMap(self):
		for (x,y) in self.graph.pheromoneMap:
			self.graph.pheromoneMap[(x,y)] = self.graph.pheromoneMap[(x,y)]*(1.0-self.evaporation)

	def insertPheromoneMap(self):
		for ant in self.ants:
			for i in range(len(ant.path)-1):
				self.graph.pheromoneMap[(ant.path[i],ant.path[i+1])] += (1/self.graph.edges[(ant.path[i],ant.path[i+1])])

	def plot(self, line_width=1, point_radius=math.sqrt(2.0), annotation_size=8, dpi=120, save=True, name=None):
		x = [self.graph[i][0] for i in self.gbest]
		x.append(x[0])
		y = [self.graph[i][1] for i in self.gbest]
		y.append(y[0])
		plt.plot(x, y, linewidth=line_width)
		plt.scatter(x, y, s=math.pi * (point_radius ** 2.0))
		plt.title('ACO')
		for i in self.gbest:
				plt.annotate(self.labels[i], self.graph[i], size=annotation_size)
		if save:
				if name is None:
						name = '{0}.png'.format('aco')
				plt.savefig(name, dpi=dpi)
		plt.show()
		plt.gcf().clear()

	def run(self):
		
		for t in range(self.iterations):
			self.ants = []
			for i in range(self.ants):
	
				initial_position = random.randint(0, self.graph.amount_vertices-1)
				ant = Ant(initial_position)
				self.ants.append(ant)
			
			# formigas se movimentam
			for i in range(self.graph.amount_vertices-1):
				for ant in self.ants:
					last_pos = ant.last_pos
					next_pos = self.calculate_next_move(ant)

					ant.path.append(next_pos)
					ant.pathCost += self.graph.edges[(last_pos,next_pos)]		
					ant.last_pos = next_pos

			# Realiza a evaporação do feromonio
			self.evaporePheromoneMap()

			# Depositar o feromonio na volta
			self.insertPheromoneMap()

			# fechando o ciclo
			for ant in self.ants:
				ant.path.append(ant.initial_pos)
				ant.pathCost += self.graph.edges[(ant.last_pos,ant.initial_pos)]

			# atualiza o melhor global 
			if(t == 0):
				self.gbest = min(self.ants, key=attrgetter('pathCost'))
			else:
				pbest = min(self.ants, key=attrgetter('pathCost'))
				if(pbest.pathCost < self.gbest.pathCost):
					self.gbest = pbest
					
			print("GBEST = ",self.gbest.path, ' - ',self.gbest.pathCost)

		# self.plot()
				
def main(it, name, length):
	graph = Graph(amount_vertices=length)
	graph.loadAssym('../data/assymetric/'+name, length)
	
	aco = ACO(graph, iterations=1000, ants=100,evaporation=0.1,alfa=1,beta=5)
	aco.run()
	return aco.gbest.pathCost

if __name__ == "__main__":	
	atsp_name = [
					"ft53",
					"ftv33",
					"ftv38",
					"ftv170",
					"kro124p",
					"rbg323",
					"rbg358",
					"rbg403",
					"rbg443"]
	lengths = [53, 34, 39, 171, 100, 323, 358, 403, 443]

	for i in range(len(atsp_name)):
		cost = 0
		best = 0
		worst = 10000000
		mean_time = 0

		f = open("results/"+atsp_name[i]+'.txt', "a")
		f.write(f'---------{atsp_name[i]} : {lengths[i]}---------\n\n')
		f.close()

		for j in range(30):
			start_time = time.time()
			cost += main(j+1, atsp_name[i], lengths[i])

			if(cost < best):
				best = cost
			elif(cost > worst):
				worst = cost
		
		final_time = time.time()
		mean_time += final_time - start_time

		f = open("results/"+atsp_name[i]+'.txt', "a")
		f.write(f'Melhor custo : {best}\n\n')
		f.write(f'Pior custo : {worst}\n\n')
		f.write(f'Media custo : {cost/30}\n\n')
		f.write(f'Tempo médio : {mean_time/30}\n\n')
		f.close()