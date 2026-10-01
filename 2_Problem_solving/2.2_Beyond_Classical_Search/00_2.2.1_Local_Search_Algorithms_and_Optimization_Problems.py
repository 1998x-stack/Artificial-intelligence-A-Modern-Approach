# 00_2.2.1_Local_Search_Algorithms_and_Optimization_Problems

"""

Lecture: 2_Problem-solving/2.2_Beyond_Classical_Search
Content: 00_2.2.1_Local_Search_Algorithms_and_Optimization_Problems

"""

import numpy as np
from typing import Callable, Tuple, Any

class LocalSearchAlgorithm:
    """
    Base class for local search algorithms.
    """
    def __init__(self, initial_state: np.ndarray, objective_function: Callable[[np.ndarray], float]):
        """
        Initialize the local search algorithm.

        Args:
        - initial_state (np.ndarray): Initial state of the algorithm.
        - objective_function (Callable[[np.ndarray], float]): Objective function to optimize.
        """
        self.current_state = initial_state
        self.objective_function = objective_function

    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Abstract method for optimizing the objective function.

        Returns:
        - Tuple[np.ndarray, float]: Optimized state and its corresponding objective function value.
        """
        raise NotImplementedError("Subclasses should implement the optimize method.")

class HillClimbing(LocalSearchAlgorithm):
    """
    Hill Climbing algorithm for local search.
    """
    def optimize(self, max_iterations: int = 1000) -> Tuple[np.ndarray, float]:
        """
        Optimize the objective function using Hill Climbing.

        Args:
        - max_iterations (int): Maximum number of iterations to perform hill climbing.

        Returns:
        - Tuple[np.ndarray, float]: Optimized state and its corresponding objective function value.
        """
        for _ in range(max_iterations):
            neighbor = self.current_state + np.random.normal(size=self.current_state.shape)
            if self.objective_function(neighbor) > self.objective_function(self.current_state):
                self.current_state = neighbor
        best_value = self.objective_function(self.current_state)
        return self.current_state, best_value

class SimulatedAnnealing(LocalSearchAlgorithm):
    """
    Simulated Annealing algorithm for local search.
    """
    def optimize(self, max_iterations: int = 1000, initial_temperature: float = 1.0, cooling_rate: float = 0.95) -> Tuple[np.ndarray, float]:
        """
        Optimize the objective function using Simulated Annealing.

        Args:
        - max_iterations (int): Maximum number of iterations.
        - initial_temperature (float): Initial temperature for Simulated Annealing.
        - cooling_rate (float): Rate at which temperature decreases.

        Returns:
        - Tuple[np.ndarray, float]: Optimized state and its corresponding objective function value.
        """
        temperature = initial_temperature
        for _ in range(max_iterations):
            neighbor = self.current_state + np.random.normal(size=self.current_state.shape)
            delta_e = self.objective_function(neighbor) - self.objective_function(self.current_state)
            if delta_e > 0 or np.random.rand() < np.exp(delta_e / temperature):
                self.current_state = neighbor
            temperature *= cooling_rate
        best_value = self.objective_function(self.current_state)
        return self.current_state, best_value

class GeneticAlgorithm:
    """
    Genetic Algorithm for optimization.
    """
    def __init__(self, population_size: int, chromosome_length: int, fitness_function: Callable[[np.ndarray], float]):
        """
        Initialize the genetic algorithm.

        Args:
        - population_size (int): Size of the population.
        - chromosome_length (int): Length of each chromosome.
        - fitness_function (Callable[[np.ndarray], float]): Fitness function to evaluate chromosomes.
        """
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.fitness_function = fitness_function
        self.population = np.random.rand(population_size, chromosome_length)

    def evolve(self, generations: int = 100) -> Tuple[np.ndarray, float]:
        """
        Evolve the population using genetic algorithm.

        Args:
        - generations (int): Number of generations to evolve.

        Returns:
        - Tuple[np.ndarray, float]: Optimized state and its corresponding objective function value.
        """
        for _ in range(generations):
            # Evaluate fitness of each individual
            fitness_values = np.array([self.fitness_function(chromosome) for chromosome in self.population])

            # Select parents based on fitness
            parents = self.population[np.argsort(fitness_values)[-2:]]

            # Crossover and mutation
            offspring = np.array([self.crossover(parents) for _ in range(self.population_size)])
            self.population = offspring

        # Find the best individual
        best_index = np.argmax([self.fitness_function(chromosome) for chromosome in self.population])
        best_individual = self.population[best_index]
        best_value = self.fitness_function(best_individual)

        return best_individual, best_value

    def crossover(self, parents: np.ndarray) -> np.ndarray:
        """
        Perform crossover between two parent chromosomes.

        Args:
        - parents (np.ndarray): Two parent chromosomes.

        Returns:
        - np.ndarray: Offspring chromosome.
        """
        crossover_point = np.random.randint(self.chromosome_length)
        offspring = np.concatenate((parents[0][:crossover_point], parents[1][crossover_point:]))
        return offspring

# Example usage:
if __name__ == "__main__":
    # Define an example objective function (minimize a quadratic function)
    def objective_function(x: np.ndarray) -> float:
        return np.sum(x**2)

    # Initial state for algorithms
    initial_state = np.random.rand(5)

    # Hill Climbing optimization
    hill_climbing = HillClimbing(initial_state, objective_function)
    optimized_state, best_value = hill_climbing.optimize()
    print(f"Hill Climbing: Optimized state = {optimized_state}, Best value = {best_value}")

    # Simulated Annealing optimization
    simulated_annealing = SimulatedAnnealing(initial_state, objective_function)
    optimized_state, best_value = simulated_annealing.optimize()
    print(f"Simulated Annealing: Optimized state = {optimized_state}, Best value = {best_value}")

    # Genetic Algorithm optimization
    genetic_algorithm = GeneticAlgorithm(population_size=10, chromosome_length=5, fitness_function=objective_function)
    optimized_state, best_value = genetic_algorithm.evolve()
    print(f"Genetic Algorithm: Optimized state = {optimized_state}, Best value = {best_value}")
