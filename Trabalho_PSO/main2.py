from operator import attrgetter, le
import random, sys, time, copy


class Graph:

	def __init__(self, amount_vertices):
		self.edges = {} 
		self.vertices = set()
		self.amount_vertices = amount_vertices 

	def addEdge(self, src, dest, cost = 0):
		if (src, dest) not in self.edges:
			self.edges[(src, dest)] = cost
			self.vertices.add(src)
			self.vertices.add(dest)

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

			# # não repetir elementos
			# if list_temp not in random_paths:
			# 	random_paths.append(list_temp)

		return random_paths
	
	def valid_path(self,path):

		for i in range(len(path)-1):
			if(path[i],path[i+1]) not in self.edges:
				print("Invalid path generated.. dissmised")
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


class Particle:

	def __init__(self, solution, cost):

		self.solution = solution

		self.pbest = solution

		self.cost_current_solution = cost
		self.cost_pbest_solution = cost

		self.velocity = []

	def setPBest(self, new_pbest):
		self.pbest = new_pbest

	def getPBest(self):
		return self.pbest

	def setVelocity(self, new_velocity):
		self.velocity = new_velocity

	def getVelocity(self):
		return self.velocity

	def setCurrentSolution(self, solution):
		self.solution = solution

	def getCurrentSolution(self):
		return self.solution

	def setCostPBest(self, cost):
		self.cost_pbest_solution = cost

	def getCostPBest(self):
		return self.cost_pbest_solution

	def setCostCurrentSolution(self, cost):
		self.cost_current_solution = cost

	def getCostCurrentSolution(self):
		return self.cost_current_solution

	def clearVelocity(self):
		del self.velocity[:]


class PSO:

	def __init__(self, graph, iterations, size_population,initial_position,cross_percent, beta=1, alfa=1):
		self.graph = graph 
		self.iterations = iterations 
		self.size_population = size_population 
		self.particles = [] 
		self.beta = beta
		self.alfa = alfa 
		self.cross_percent = cross_percent 

		# iniciar a população 
		solutions = self.graph.getRandomPaths(self.size_population,initial_position)

		for solution in solutions:
			
			particle = Particle(solution=solution, cost=graph.getCostPath(solution))
			
			self.particles.append(particle)

		
		self.size_population = len(self.particles)


	def getGBest(self):
		return self.gbest


	def showsParticles(self):

		print('Partículas do PSO...\n')
		for particle in self.particles:
			print('Melhor Local: %s\t|\tCusto: %d\t|\tPath: %s\t|\tCusto Atual: %d' \
				% (str(particle.getPBest()), particle.getCostPBest(), str(particle.getCurrentSolution()),
							particle.getCostCurrentSolution()))
		print('')

	def order_cx(self, p1, p2):

		a = random.randint(0,self.graph.amount_vertices-2)
		b = random.randint(a+1,self.graph.amount_vertices-1)

		f1, f2 = [0]*self.graph.amount_vertices, [0]*self.graph.amount_vertices

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

	def pmx_cx(self, p1, p2):

		a = random.randint(0,self.graph.amount_vertices-2)
		b = random.randint(a+1,self.graph.amount_vertices-1)

		f1, f2 = [0]*self.graph.amount_vertices, [0]*self.graph.amount_vertices

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

	def cycle_cx(self, p1, p2):

		f1, f2 = [0]*self.graph.amount_vertices, [0]*self.graph.amount_vertices

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

	def run(self):
		

		for t in range(self.iterations):

			# atualiza o melhor global 
			self.gbest = min(self.particles, key=attrgetter('cost_pbest_solution'))
			
			for particle in self.particles:
				
				particle.clearVelocity()
				temp_velocity = []
				solution_gbest = copy.copy(self.gbest.getPBest()) 
				solution_pbest = particle.getPBest()[:] 
				solution_particle = particle.getCurrentSolution()[:] 

				# calcula os swaps pra calcular o melhor local
				for i in range(self.graph.amount_vertices):
					if solution_particle[i] != solution_pbest[i]:
						
						swap_operator = (i, solution_pbest.index(solution_particle[i]), self.alfa)

						temp_velocity.append(swap_operator)

						aux = solution_pbest[swap_operator[0]]
						solution_pbest[swap_operator[0]] = solution_pbest[swap_operator[1]]
						solution_pbest[swap_operator[1]] = aux

				# calcula os swaps pra calcular o melhor global
				for i in range(self.graph.amount_vertices):
					if solution_particle[i] != solution_gbest[i]:
						
						swap_operator = (i, solution_gbest.index(solution_particle[i]), self.beta)

						temp_velocity.append(swap_operator)

						aux = solution_gbest[swap_operator[0]]
						solution_gbest[swap_operator[0]] = solution_gbest[swap_operator[1]]
						solution_gbest[swap_operator[1]] = aux

				
				particle.setVelocity(temp_velocity)

				# faz a trocas e gera o a nova solução da particula
				for swap_operator in temp_velocity:
					if random.random() <= swap_operator[2]:
						
						aux = solution_particle[swap_operator[0]]
						solution_particle[swap_operator[0]] = solution_particle[swap_operator[1]]
						solution_particle[swap_operator[1]] = aux
				
				particle.setCurrentSolution(solution_particle)
				
				cost_current_solution = self.graph.getCostPath(solution_particle)
				
				# updates the cost of the current solution
				particle.setCostCurrentSolution(cost_current_solution)

				# checar se a solucao atual é a melhor local
				if cost_current_solution < particle.getCostPBest():
					particle.setPBest(solution_particle)
					particle.setCostPBest(cost_current_solution)

			# aplicar the oxc
			for c in range(int((self.cross_percent*self.size_population)/2)):

				p1 = random.randint(0,self.size_population-1)
				p2 = random.randint(0,self.size_population-1)

				sol_f1,sol_f2 = self.cycle_cx(self.particles[p1].solution,self.particles[p2].solution)

				f1 = Particle(sol_f1,self.graph.getCostPath(sol_f1))
				f2 = Particle(sol_f2,self.graph.getCostPath(sol_f2))

				self.particles.append(f1)
				self.particles.append(f2)

			# Ordenar as particulas pelo fitness
			self.particles.sort(key=Particle.getCostCurrentSolution)

			# Reset the original size of population
			self.particles = self.particles[:self.size_population]

			


		

if __name__ == "__main__":
	
	
	graph = Graph(amount_vertices=6)	

	graph.loadfri06()

	# creates a PSO instance
	pso = PSO(graph, iterations=500, size_population=150,initial_position=3,cross_percent=0.8, beta=1, alfa=0.9)

	print('População inicial : ')
	pso.showsParticles()
	print()
	pso.run() # runs the PSO algorithm
	pso.showsParticles() # shows the particles

	pmxFile = open("cicle_results.txt","a")
	# shows the global best particle
	print('gbest: %s | cost: %d\n' % (pso.getGBest().getPBest(), pso.getGBest().getCostPBest()))
	pmxFile.write(str(pso.getGBest().getCostPBest())+" \n")