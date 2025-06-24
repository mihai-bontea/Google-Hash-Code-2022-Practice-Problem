#pragma once
#include <memory>
#include <bitset>

#include "Defines.h"

struct Individual
{
	int fitness;
	std::unique_ptr<std::bitset<MAX_INGREDIENTS>> genes;

	Individual()
		: genes(std::make_unique<std::bitset<MAX_INGREDIENTS>>())
		, fitness(0)
	{}

    Individual(const Individual& other)
        : fitness(other.fitness)
        , genes(std::make_unique<std::bitset<MAX_INGREDIENTS>>(*other.genes))
    {}

    Individual& operator=(const Individual& other)
    {
        if (this != &other)
        {
            fitness = other.fitness;
            *genes = *other.genes;
        }
        return *this;
    }

    Individual(Individual&&) = default;

    Individual& operator=(Individual&&) = default;
};