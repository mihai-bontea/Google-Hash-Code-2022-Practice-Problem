#pragma once
#include <memory>
#include <bitset>

#include "Defines.h"

struct Individual
{
	int fitness;
	std::unique_ptr<std::bitset<MAX_INGREDIENTS>> genes; // is_ingredient_chosen

	Individual()
		: genes(std::make_unique<std::bitset<MAX_INGREDIENTS>>())
		, fitness(0)
	{
	}
};