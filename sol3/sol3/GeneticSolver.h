#pragma once

#include <array>
#include <memory>
#include <bitset>
#include <random>
#include <chrono>
#include "Data.h"

#include <omp.h>
#include "FitnessEvaluator.cuh"
#include "BoundedPriorityQueue.h"

struct MinHeapComp {
	bool operator()(const Individual& lhs, const Individual& rhs) const {
		return lhs.fitness > rhs.fitness;
	}
};

class GeneticSolver
{
private:
	const Data& data;
	std::mt19937 rng;
	std::uniform_real_distribution<double> real_dist;
	const std::chrono::steady_clock::time_point start;

	//std::vector<Individual> population;
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
		Individual ind;
		for (int i = 0; i < data.ingredients.size(); ++i)
			ind.genes.get()[i] = (real_dist(rng) < 0.5);
		return ind;
	}

	void initialize_population()
	{
		#pragma omp parallel
		for (int ingredient_index = 0; ingredient_index < POPULATION_SIZE; ++ingredient_index)
		{
			population[ingredient_index] = get_random_individual();
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

	Individual get_best_individual()
	{
		initialize_population();
		Individual best_individual;

		size_t generation = 0;
		BoundedPriorityQueue<Individual, MinHeapComp> top_n_queue(10 * POPULATION_SIZE / 100);

		while (!is_timer_expired())
		{
			// Evaluate the fitness of the current generation
			fitness_evaluator.get()->evaluate();

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

			// Choose the rest of the individuals by tournament selection
			while (index < POPULATION_SIZE)
			{
				const Individual& parent1 = tournament_select();
				const Individual& parent2 = tournament_select();

				Individual child = crossover(parent1, parent2);
				mutate(child);

				new_population[index++] = std::move(child);
			}

			// Replace the old generation
			population = std::move(new_population);
			++generation;
		}
		return best_individual;
	}

public:
	GeneticSolver(const Data& data)
		: real_dist(0.0, 1.0)
		, rng(static_cast<unsigned>(std::time(nullptr)))
		, data(data)
		, start(std::chrono::steady_clock::now())
		, fitness_evaluator(std::make_unique<CpuFitnessEvaluator>(data, population))
	{
		//population.reserve(POPULATION_SIZE);
	}

	std::vector<std::string> solve()
	{
		const auto best_individual = get_best_individual();

		std::vector<std::string> result(data.universally_liked.begin(), data.universally_liked.end());

		std::cout << "Score = " << best_individual.fitness << "\n\n";

		for (int ingredient_index = 0; ingredient_index < data.ingredients.size(); ++ingredient_index)
			if ((*best_individual.genes)[ingredient_index])
				result.push_back(data.ingredients[ingredient_index]);
		return result;
	}
};