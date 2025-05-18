#pragma once

#include <array>
#include <random>
#include <chrono>
#include <iostream>

#include "SimulationState.h"
#include "BoundedPriorityQueue.h"

#define NR_THREADS 10

struct MaxHeapComp {
    bool operator()(const SimulationState& lhs, const SimulationState& rhs) const {
        return lhs.clients_lost < rhs.clients_lost;
    }
};

class HybridSimulatedAnnealing
{
    const Data &data;
    const std::chrono::steady_clock::time_point start;

    bool is_timer_expired()
    {
        const auto now = std::chrono::steady_clock::now();
        auto elapsed = duration_cast<std::chrono::minutes>(now - start);
        return elapsed.count() >= 28;
    }

    std::vector<int> get_client_satisfaction(const SimulationState& simulation_state) const
    {
        std::vector<int> client_satisfaction(data.client_to_satisfaction_req.begin(), data.client_to_satisfaction_req.end());
        for (int ingredient_index = 0; ingredient_index < data.ingredients.size(); ++ingredient_index)
        {
            const auto &ingredient = data.ingredients[ingredient_index];
            const auto ingr_fans_it = data.ingr_to_fans.find(ingredient);
            const auto ingr_haters_it = data.ingr_to_haters.find(ingredient);

            if ((*simulation_state.is_ingredient_chosen)[ingredient_index])
            {
                // Decrease satisfaction for clients who dislike this ingredient
                if (ingr_haters_it != data.ingr_to_haters.end())
                    for (int client_id: ingr_haters_it->second)
                        client_satisfaction[client_id]--;
            }
            else
            {
                // Decrease satisfaction for clients who like this ingredient
                if (ingr_fans_it != data.ingr_to_fans.end())
                    for (int client_id: ingr_fans_it->second)
                        client_satisfaction[client_id]--;
            }
        }
        return client_satisfaction;
    }

    int get_nr_clients_won_by_removal(const std::vector<int> &client_satisfaction, int ingredient_index) const
    {
        const auto &ingredient = data.ingredients[ingredient_index];
        const auto ingr_fans_it = data.ingr_to_fans.find(ingredient);
        const auto ingr_haters_it = data.ingr_to_haters.find(ingredient);

        // Check how many clients we win by removing the ingredient
        int gain = 0;
        if (ingr_haters_it != data.ingr_to_haters.end())
            for (int client_id: ingr_haters_it->second)
                if (client_satisfaction[client_id] == data.client_to_satisfaction_req[client_id] - 1)
                    gain++;

        // Check how many clients we lose by removing the ingredient
        int loss = 0;
        if (ingr_fans_it != data.ingr_to_fans.end())
            for (int client_id: ingr_fans_it->second)
                if (client_satisfaction[client_id] == data.client_to_satisfaction_req[client_id])
                    loss++;

        return gain - loss;
    }

    void apply_removal(std::vector<int> &client_satisfaction, int ingredient_index) const
    {
        const auto &ingredient = data.ingredients[ingredient_index];
        const auto ingr_fans_it = data.ingr_to_fans.find(ingredient);
        const auto ingr_haters_it = data.ingr_to_haters.find(ingredient);

        // Increase the satisfaction of clients who dislike this ingredient
        if (ingr_haters_it != data.ingr_to_haters.end())
            for (int client_id: ingr_haters_it->second)
                client_satisfaction[client_id]++;

        // Decrease the satisfaction of clients who like this ingredient
        if (ingr_fans_it != data.ingr_to_fans.end())
            for (int client_id: ingr_fans_it->second)
                client_satisfaction[client_id]--;
    }

    int get_nr_clients_won_by_addition(const std::vector<int> &client_satisfaction, int ingredient_index) const
    {
        const auto &ingredient = data.ingredients[ingredient_index];
        const auto ingr_fans_it = data.ingr_to_fans.find(ingredient);
        const auto ingr_haters_it = data.ingr_to_haters.find(ingredient);

        // Check how many clients we win by adding the ingredient
        int gain = 0;
        if (ingr_fans_it != data.ingr_to_fans.end())
            for (int client_id: ingr_fans_it->second)
                if (client_satisfaction[client_id] == data.client_to_satisfaction_req[client_id] - 1)
                    gain++;

        // Check how many clients we lose by adding the ingredient
        int loss = 0;
        if (ingr_haters_it != data.ingr_to_haters.end())
            for (int client_id: ingr_haters_it->second)
                if (client_satisfaction[client_id] == data.client_to_satisfaction_req[client_id])
                    loss++;

        return gain - loss;
    }

    void apply_addition(std::vector<int> &client_satisfaction, int ingredient_index) const
    {
        const auto &ingredient = data.ingredients[ingredient_index];
        const auto ingr_fans_it = data.ingr_to_fans.find(ingredient);
        const auto ingr_haters_it = data.ingr_to_haters.find(ingredient);

        // Increase the satisfaction of clients who like this ingredient
        if (ingr_fans_it != data.ingr_to_fans.end())
            for (int client_id: ingr_fans_it->second)
                client_satisfaction[client_id]++;

        // Decrease the satisfaction of clients who dislike this ingredient
        if (ingr_haters_it != data.ingr_to_haters.end())
            for (int client_id: ingr_haters_it->second)
                client_satisfaction[client_id]--;
    }

    SimulationState simulated_annealing(SimulationState initial_state)
    {
        auto rng = std::mt19937(std::random_device{}());
        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        std::uniform_int_distribution<int> ingredient_dist(0, data.ingredients.size() - 1);

        SimulationState current_state = std::move(initial_state);
        SimulationState best_simulation_state = current_state;

        auto current_satisfaction = get_client_satisfaction(current_state);

        double temperature = 100.0;
        const double cooling_rate = 0.999;
        const double absolute_temp = 1e-3;

        while (temperature > absolute_temp)
        {
            // Pick a random ingredient to flip
            int ingredient_index = ingredient_dist(rng);
            bool was_chosen = (*current_state.is_ingredient_chosen)[ingredient_index];

            int delta;
            if (was_chosen)
                delta = get_nr_clients_won_by_removal(current_satisfaction, ingredient_index) * -1;
            else
                delta = get_nr_clients_won_by_addition(current_satisfaction, ingredient_index) * -1;

            if (delta <= 0 || prob_dist(rng) < std::exp(-delta / temperature))
            {
                if (was_chosen)
                    apply_removal(current_satisfaction, ingredient_index);
                else
                    apply_addition(current_satisfaction, ingredient_index);
                (*current_state.is_ingredient_chosen)[ingredient_index] = !was_chosen;
                current_state.clients_lost += delta;

                if (current_state.clients_lost < best_simulation_state.clients_lost)
                {
                    best_simulation_state = current_state;
                }
            }
            temperature *= cooling_rate;
        }
        return best_simulation_state;
    }

    void local_search(SimulationState& simulation_state)
    {
        // Get the client satisfaction for the best solution found by the simulation
        auto client_satisfaction = get_client_satisfaction(simulation_state);
        int clients_gained = 0;

        bool improved = true;
        while (improved)
        {
            improved = false;

            for (int ingredient_index = 0; ingredient_index < data.ingredients.size(); ++ingredient_index)
                if ((*simulation_state.is_ingredient_chosen)[ingredient_index])
                {
                    const int removal_gain = get_nr_clients_won_by_removal(client_satisfaction, ingredient_index);
                    if (removal_gain > 0)
                    {
                        (*simulation_state.is_ingredient_chosen)[ingredient_index] = false;
                        apply_removal(client_satisfaction, ingredient_index);

                        improved = true;
                        clients_gained += removal_gain;
                    }
                }
                else
                {
                    const int addition_gain = get_nr_clients_won_by_addition(client_satisfaction, ingredient_index);
                    if (addition_gain > 0)
                    {
                        (*simulation_state.is_ingredient_chosen)[ingredient_index] = true;
                        apply_addition(client_satisfaction, ingredient_index);

                        improved = true;
                        clients_gained += addition_gain;
                    }
                }
        }
    }

public:
    explicit HybridSimulatedAnnealing(const Data& data)
            : data(data)
            , start(std::chrono::steady_clock::now())
    {}

    SimulationState attempt_improvement(SimulationState starting_state)
    {
        // Obtain some starting states
        BoundedPriorityQueue<SimulationState, MaxHeapComp> top_n_queue(NR_THREADS);

        #pragma omp parallel for
        for (int th_index = 0; th_index < NR_THREADS; ++th_index)
        {
            auto simulation_state = simulated_annealing(starting_state);
            local_search(simulation_state);

            #pragma omp critical
            {
                top_n_queue.push(simulation_state);
            }
        }
        SimulationState best_simulation_state = starting_state;
        while (!is_timer_expired())
        {
            auto best_n_states = top_n_queue.extract_sorted();

            #pragma omp parallel for
            for (int th_index = 0; th_index < best_n_states.size(); ++ th_index)
            {
                auto simulation_state = simulated_annealing(best_n_states[th_index]);
                local_search(simulation_state);

                #pragma omp critical
                {
                    top_n_queue.push(simulation_state);
                    if (best_simulation_state.clients_lost > simulation_state.clients_lost)
                    {
                        best_simulation_state = simulation_state;
                        std::cout << "New best state has " << best_simulation_state.clients_lost << " clients lost. [SA]\n";
                    }
                }
            }

            // Reinsert the previous states if they are better
            for (const auto& simulation_state : best_n_states)
                top_n_queue.push(simulation_state);
        }
        return best_simulation_state;
    }
};