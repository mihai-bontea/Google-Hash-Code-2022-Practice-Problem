#pragma once

#include <array>
#include <memory>
#include <bitset>
#include <random>
#include <chrono>
#include "Data.h"

#include <omp.h>


struct Individual
{
	int fitness;
	std::unique_ptr<std::bitset<MAX_CLIENTS>> genes; // is_client_remaining

	Individual()
	: genes(std::make_unique<std::bitset<MAX_CLIENTS>>())
	, fitness(0)
	{}
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

	void grade_fitness()
	{

	}

	void initialize_population()
	{
		#pragma omp parallel
		for (int i = 0; i < POPULATION_SIZE; ++i)
		{
			population[i] = get_random_individual();
		}
	}

	void mutate(Individual& ind)
	{
		for (int i = 0; i < data.ingredients.size(); ++i)
		{
			if (real_dist(rng) < MUTATION_RATE)
				ind.genes.get()->flip(i);
		}
	}

public:
	GeneticSolver(const Data& data)
		: real_dist(0.0, 1.0)
		, rng(static_cast<unsigned>(std::time(nullptr)))
		, data(data)
		, start(std::chrono::steady_clock::now())
	{
		//population.reserve(POPULATION_SIZE);
	}

	void get_best_solution(int minutes)
	{
		initialize_population();

		size_t generation = 0;
		while (!is_timer_expired())
		{

		}
	}

};