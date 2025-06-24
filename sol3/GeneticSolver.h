#pragma once

#include <omp.h>
#include <array>
#include <memory>
#include <bitset>
#include <random>
#include <chrono>
#include <unordered_set>

#include "Data.h"
#include "FitnessEvaluator.cuh"
#include "SimulatedAnnealingParallel.h"

class GeneticSolver
{
private:
	const Data& data;
	std::mt19937 rng;
	std::uniform_real_distribution<double> real_dist;
	std::uniform_int_distribution<int> individual_dist;
	const std::chrono::steady_clock::time_point start;

	std::array<Individual, POPULATION_SIZE> population;
	std::unique_ptr<IFitnessEvaluator> fitness_evaluator;

	bool is_timer_expired()
	{
		const auto now = std::chrono::steady_clock::now();
		auto elapsed = duration_cast<std::chrono::minutes>(now - start);
		return elapsed.count() >= SIMULATION_LENGTH_MINUTES;
	}

	Individual get_random_individual()
	{
		Individual individual;
		for (int index = 0; index < data.nr_ingredients; ++index)
			(*individual.genes)[index] = (real_dist(rng) < 0.5);
		return individual;
	}

	void initialize_population()
	{
		for (int index = 0; index < POPULATION_SIZE; ++index)
		{
			population[index] = get_random_individual();
		}
	}

	void mutate(Individual& ind)
	{
		for (int index = 0; index < data.ingredients.size(); ++index)
		{
			if (real_dist(rng) < MUTATION_RATE)
				ind.genes.get()->flip(index);
		}
	}

	Individual& tournament_select(int tournament_size = 3)
	{
		int best_index = rng() % POPULATION_SIZE;
		for (int attempt = 1; attempt < tournament_size; ++attempt)
		{
			int index = rng() % POPULATION_SIZE;
			if (population[index].fitness > population[best_index].fitness)
				best_index = index;
		}
		return population[best_index];
	}

	Individual crossover(const Individual& parent1, const Individual& parent2)
	{
		Individual child;
		for (int index = 0; index < data.ingredients.size(); ++index)
		{
			bool gene = (rng() % 2 == 0) ? (*parent1.genes)[index] : (*parent2.genes)[index];
			child.genes->set(index, gene);
		}
		return child;
	}

	std::vector<int> get_random_individual_indices()
	{
		std::unordered_set<int> selected_indices;

		while (selected_indices.size() < (10 * POPULATION_SIZE / 100))
		{
			int index = individual_dist(rng);
			selected_indices.insert(index);
		}

		return std::vector(selected_indices.begin(), selected_indices.end());
	}

	void improve_ten_percent()
	{
		const auto indices_to_modify = get_random_individual_indices();

		#pragma omp parallel for num_threads(12)
		for (int index = 0; index < indices_to_modify.size(); ++index)
		{
			SimulatedAnnealingParallel simulated_annealing(data, 1);
			population[indices_to_modify[index]] = std::move(simulated_annealing.attempt_improvement(population[indices_to_modify[index]]));
		}
	}

	Individual get_best_individual()
	{
		std::cout << "About to initialize the population\n";
		initialize_population();
		Individual best_individual;

		size_t generation = 0;
		BoundedPriorityQueue<Individual, MinHeapComp> top_n_queue(10 * POPULATION_SIZE / 100);

		while (!is_timer_expired())
		{
			std::cout << "Starting generation " << generation << std::endl;

			// Improve 10% of the population with simulated annealing and local search every 1500 generations
			if (generation % 1500 == 0)
				improve_ten_percent();

			// Evaluate the fitness of the current generation
			fitness_evaluator.get()->evaluate();

			std::cout << "Successfully evaluated.\n";

			// Save the top 10% of individuals
			for (const auto& individual : population)
				top_n_queue.push(individual);

			std::array<Individual, POPULATION_SIZE> new_population;
			int index = 0;

			for (const auto& individual : top_n_queue.extract_sorted())
			{
				new_population[index++] = individual;
			}
			// The first extracted has the highest fitness
			best_individual = new_population[0];
			std::cout << "The best individual so far has fitness = " << best_individual.fitness << std::endl;;

			// Choose the rest of the individuals by tournament selection
			while (index < POPULATION_SIZE)
			{
				const Individual& parent1 = tournament_select();
				const Individual& parent2 = tournament_select();

				Individual child = crossover(parent1, parent2);
				mutate(child);

				new_population[index++] = std::move(child);
			}

			std::cout << "Built the new generation\n";

			// Replace the old generation
			population = std::move(new_population);
			++generation;

			std::cout << "\n\n";
		}
		return best_individual;
	}

public:
	GeneticSolver(const Data& data)
		: real_dist(0.0, 1.0)
		, individual_dist(0, POPULATION_SIZE - 1)
		, rng(static_cast<unsigned>(std::time(nullptr)))
		, data(data)
		, start(std::chrono::steady_clock::now())
		, fitness_evaluator(std::make_unique<CpuFitnessEvaluator>(data, population))
	{}

	std::vector<std::string> solve()
	{
		const auto best_individual = get_best_individual();

		// Apply simulated annealing and local search to the best solution found so far
		SimulatedAnnealingParallel simulated_annealing(data, 5);
		const auto improved_individual = simulated_annealing.attempt_improvement_parallel(best_individual);

		std::vector<std::string> result(data.universally_liked.begin(), data.universally_liked.end());

		std::cout << "Score = " << best_individual.fitness << "\n\n";

		for (int ingredient_index = 0; ingredient_index < data.ingredients.size(); ++ingredient_index)
			if ((*improved_individual.genes)[ingredient_index])
				result.push_back(data.ingredients[ingredient_index]);
		return result;
	}
};