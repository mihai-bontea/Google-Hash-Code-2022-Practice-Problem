#pragma once

#include <bitset>
#include <memory>

#include "Data.h"

struct SimulationState
{
    int depth = 0, clients_lost = 0;
    std::unique_ptr<std::bitset<MAX_CLIENTS>> is_client_remaining;
    std::unique_ptr<std::bitset<MAX_INGREDIENTS>> is_ingredient_chosen;

    SimulationState()
            : is_client_remaining(std::make_unique<std::bitset<MAX_CLIENTS>>()),
              is_ingredient_chosen(std::make_unique<std::bitset<MAX_INGREDIENTS>>())
    {
        is_client_remaining->set();
        is_ingredient_chosen->reset();
    }

    SimulationState(const SimulationState &other)
            : depth(other.depth), clients_lost(other.clients_lost)
            , is_client_remaining(std::make_unique<std::bitset<MAX_CLIENTS>>(*other.is_client_remaining))
            , is_ingredient_chosen(std::make_unique<std::bitset<MAX_INGREDIENTS>>(*other.is_ingredient_chosen))
    {}

    SimulationState &operator=(const SimulationState &other)
    {
        if (this != &other)
        {
            depth = other.depth;
            clients_lost = other.clients_lost;
            *is_client_remaining = *other.is_client_remaining;
            *is_ingredient_chosen = *other.is_ingredient_chosen;
        }
        return *this;
    }

    SimulationState(SimulationState &&) = default;

    SimulationState &operator=(SimulationState &&) = default;
};