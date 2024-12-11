import numpy as np
from scipy.ndimage import label
from PIL import Image

def initialize_population(pop_size, rows, cols):
    """初始化一組隨機的染色體(2D 矩陣)"""
    return [np.random.choice([0, 2], size=(rows, cols)) for _ in range(pop_size)]

def crossover(parent1, parent2):
    """對於每個element,隨機選擇其中一個parent對應的element"""
    mask = np.random.choice([True, False], size=parent1.shape)
    child = np.where(mask, parent1, parent2)
    return child

def mutate(chromosome, mutation_rate):
    """讓每個element都有機率從0變成2或從2變成0"""
    mutation_mask = np.random.rand(*chromosome.shape) < mutation_rate
    chromosome[mutation_mask] = 2 - chromosome[mutation_mask]  # 0 -> 2 或 2 -> 0
    return chromosome

def fitness(chromosome, rows, cols):
    """fitness：計算被包起來的最大草地範圍，這個範圍越大越好"""
    grass = chromosome == 2 
    labeled, num_features = label(grass)  # 標記每片草地
    fitness_value = 0

    for region in range(1, num_features + 1):
        region_mask = labeled == region
        # 檢查該片草地是否被山完整包圍(邊緣不算山)
        found = True        
        for x in range(rows):
            for y in range(cols):
                if region_mask[x][y]==1:
                    if x==0 or x==rows-1 or y==0 or y==cols-1:
                        found = False
                        break
            if found == False:
                break
        if found:
            if fitness_value < np.sum(region_mask):
                fitness_value = np.sum(region_mask) #如果有，計算被包起來的最大草地範圍，這個範圍越大越好

    return fitness_value

def tournament_selection(population, rows, cols, k=2):
    """使用Tournament Selection來選擇parent"""
    competitors = np.random.choice(len(population), size=k, replace=False)
    fitness_scores = [fitness(population[index], rows, cols) for index in competitors]
    return population[competitors[np.argmax(fitness_scores)]]

def evolve(population, rows, cols, generations, mutation_rate, k=2):
    """GA: 使用 Tournament Selection、自訂義的crossover和fitness function"""
    for generation in range(generations):
        fitness_scores = [fitness(chromosome, rows, cols) for chromosome in population]

        next_population = []
        while len(next_population) < len(population):
            parent1 = tournament_selection(population,rows, cols, k)
            parent2 = tournament_selection(population,rows, cols, k)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            next_population.append(child)

        population = next_population

        best_fitness = max(fitness_scores)
        #print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")

    best_index = np.argmax([fitness(chromosome, rows, cols) for chromosome in population])
    return population[best_index]


def visualize_chromosome(chromosome, rows, cols, filename):
    bg = Image.new('RGB',(rows*50, cols*50), '#000000')
    img0=Image.open('./data/mountain.png').resize((50,50))
    img2=Image.open('./data/grass.png').resize((50,50))
    for i in range(rows):
        for j in range(cols):
            if chromosome[i][j]==0:
                bg.paste(img0,(i*50,j*50))
            else:
                bg.paste(img2,(i*50,j*50))
    bg.save(filename)
# 參數設置
population_size = 20
rows, cols = 10, 10
generations = 1000
mutation_rate = 0.01
for i in range(10):
    population = initialize_population(population_size, rows, cols)
    #print(population)
    best_chromosome = evolve(population, rows, cols, generations, mutation_rate, k=2)
    visualize_chromosome(best_chromosome, rows, cols, "./output/best_landscape"+str(i+1)+".png")
