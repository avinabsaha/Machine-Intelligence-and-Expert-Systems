# Name: Avinab Saha
# Roll: 15EC10071
# Mies Coding Assignment - Genetic Algorithm


import random 
import nltk
# Defining Population length, target string, valid genes
population_length = 100
target_string = "Avinab Saha 15EC10071"
valid_genes = """abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890, .-;:_!"#%&/()=?@${[]}"""

class member():
     
    def __init__(self, chromosome): 
        self.chromosome = chromosome  
        self.fitness = self.find_fitness() 
  
    
    def mutated_valid_genes(self): 
        gene = random.choice(valid_genes) 
        return gene 
  
    @classmethod
    def create_gnome(self): 
        return [random.choice(valid_genes)  for loop3 in range(len(target_string))] 
  
    def mating_process(self, par2): 
        child_chromosome = [] 
        for gp1, gp2 in zip(self.chromosome, par2.chromosome):     
            # Assign some random probability   
            prob = random.random() 
            if prob < 0.50: 
                child_chromosome.append(gp1) 
            elif prob < 0.80: 
                child_chromosome.append(gp2) 
            else: 
                child_chromosome.append(self.mutated_valid_genes()) 
        return member(child_chromosome) 
  
    def find_fitness(self):  
        return nltk.edit_distance(self.chromosome,target_string)

# Current generation 
generation = 1
  
found = False
population = [] 
  
# create initial population 
for loop in range(population_length): 
    gnome = member.create_gnome() 
    population.append(member(gnome)) 
  
while not found: 
  
    # sort the population in increasing order of fitness score 
    population = sorted(population, key = lambda x:x.fitness) 
  
    # Breaking condition
    if population[0].fitness <= 0: 
        found = True
        print("Generation ID:{}".format(generation)) 
	print("Target Value:{}".format("".join(population[0].chromosome)))
	print("Fitness Value:{}".format(population[0].fitness)) 
        break
  
    # Generate new offsprings for new generation 
    new_generation = [] 
  
    # 20% of fittest population goes to the next generation 
    s = int((20*population_length)/100) 
    new_generation.extend(population[:s]) 
  
    # From 50% of fittest population, members will mating_process to produce offspring 
    s = int((80*population_length)/100) 
    for _ in range(s): 
        parent1 = random.choice(population[:50]) 
        parent2 = random.choice(population[:50]) 
        child = parent1.mating_process(parent2) 
        new_generation.append(child) 
  
    population = new_generation 
    

    print("Generation ID:{}".format(generation)) 
    print("Target Value:{}".format("".join(population[0].chromosome)))
    print("Fitness Value:{}".format(population[0].fitness))
    generation = generation+1 
