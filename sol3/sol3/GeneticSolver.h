#pragma once

#include <memory>
#include <bitset>
#include <random>

#include "Data.h"

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

	Individual get_random_individual()
	{
		Individual ind;
		for (int i = 0; i < data.ingredients.size(); ++i)
			ind.genes.get()[i] = (real_dist(rng) < 0.5);
		return ind;
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
	{

	}

};