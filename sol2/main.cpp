#include <map>
#include <array>
#include <cmath>
#include <bitset>
#include <iostream>
#include <functional>

#include <omp.h>

#include "SimulationState.h"
#include "SimulatedAnnealing.h"

class Solver
{
private:
    const Data &data;
    SimulationState best_simulation_state;
    const std::chrono::steady_clock::time_point start;

    SimulationState update_for_ingr_addition(const SimulationState &simulation_state, int ingredient_index)
    {
        SimulationState updated_state = simulation_state;
        updated_state.depth++;
        (*updated_state.is_ingredient_chosen)[ingredient_index] = true;

        const auto &ingredient = data.ingredients[ingredient_index];
        const auto clients_it = data.ingr_to_haters.find(ingredient);

        // Nobody dislikes this ingredient
        if (clients_it == data.ingr_to_haters.end())
            return updated_state;

        // All clients who dislike this ingredient are lost
        for (int client_index: clients_it->second)
        {
            if ((*updated_state.is_client_remaining)[client_index])
            {
                (*updated_state.is_client_remaining)[client_index] = false;
                updated_state.clients_lost++;
            }
        }

        return updated_state;
    }

    SimulationState update_for_ingr_removal(const SimulationState &simulation_state, int ingredient_index)
    {
        SimulationState updated_state = simulation_state;
        updated_state.depth++;

        const auto &ingredient = data.ingredients[ingredient_index];
        const auto clients_it = data.ingr_to_fans.find(ingredient);

        // Nobody likes this ingredient
        if (clients_it == data.ingr_to_fans.end())
            return updated_state;

        // All clients who like this ingredient are lost
        for (int client_index: clients_it->second)
        {
            if ((*updated_state.is_client_remaining)[client_index])
            {
                (*updated_state.is_client_remaining)[client_index] = false;
                updated_state.clients_lost++;
            }
        }

        return updated_state;
    }

    bool is_timer_expired()
    {
        const auto now = std::chrono::steady_clock::now();
        auto elapsed = duration_cast<std::chrono::minutes>(now - start);
        return elapsed.count() >= 2;
    }

    std::vector<SimulationState> get_states_at_depth(const int target_depth)
    {
        // There are 2^(depth - 1) nodes(states)
        std::vector<SimulationState> states;
        states.reserve((size_t) pow(2, target_depth - 1));

        std::function<void(SimulationState)> simulate_to_depth = [&](SimulationState simulation_state) {
            if (simulation_state.depth == target_depth)
            {
                states.push_back(simulation_state);
            } else
            {
                simulate_to_depth(update_for_ingr_addition(simulation_state, simulation_state.depth));
                simulate_to_depth(update_for_ingr_removal(simulation_state, simulation_state.depth));
            }
        };
        simulate_to_depth(SimulationState());
        return states;
    }

    inline bool should_prune_branch(const SimulationState &simulation_state, int local_clients_lost)
    {
        return (simulation_state.clients_lost >= local_clients_lost);
    }

    void simulate(SimulationState simulation_state, int ingredient_index, int local_clients_lost)
    {
        // Solution reached
        if (ingredient_index == data.ingredients.size())
        {
            #pragma omp critical
            {
                if (simulation_state.clients_lost < best_simulation_state.clients_lost)
                {
                    best_simulation_state = simulation_state;
                    std::cout << "New best state has " << best_simulation_state.clients_lost << " clients lost. [SIM]\n";
                }
            }
            return;
        }

        // Prune branch if already worse than the last synced best solution found so far
        if (should_prune_branch(simulation_state, local_clients_lost))
            return;

        // Check the updated best solution found so far
        #pragma omp critical
        {
            local_clients_lost = best_simulation_state.clients_lost;
        }
        if (should_prune_branch(simulation_state, local_clients_lost) || is_timer_expired())
            return;

        const auto &ingredient = data.ingredients[ingredient_index];
        const auto ingr_fans_it = data.ingr_to_fans.find(ingredient);
        const auto ingr_haters_it = data.ingr_to_haters.find(ingredient);

        // No one dislikes this ingredient, no point checking for its removal
        if (ingr_haters_it == data.ingr_to_haters.end())
            simulate(update_for_ingr_addition(simulation_state, ingredient_index), ingredient_index + 1,
                     local_clients_lost);
            // No one likes this ingredient, no point checking for its addition
        else if (ingr_fans_it == data.ingr_to_fans.end())
            simulate(update_for_ingr_removal(simulation_state, ingredient_index), ingredient_index + 1,
                     local_clients_lost);

        else
        {
            SimulationState state_for_addition = update_for_ingr_addition(simulation_state, ingredient_index);
            const int clients_lost_addition = state_for_addition.clients_lost;

            SimulationState state_for_removal = update_for_ingr_removal(simulation_state, ingredient_index);
            const int clients_lost_removal = state_for_removal.clients_lost;

            if (state_for_addition.clients_lost < state_for_removal.clients_lost)
            {
                simulate(std::move(state_for_addition), ingredient_index + 1, local_clients_lost);
                if (clients_lost_addition != simulation_state.clients_lost)
                    simulate(std::move(state_for_removal), ingredient_index + 1, local_clients_lost);
            } else
            {
                simulate(std::move(state_for_removal), ingredient_index + 1, local_clients_lost);
                if (clients_lost_removal != simulation_state.clients_lost)
                    simulate(std::move(state_for_addition), ingredient_index + 1, local_clients_lost);
            }
        }
    }

public:
    explicit Solver(const Data &data)
            : data(data), start(std::chrono::steady_clock::now())
    {
        best_simulation_state.clients_lost = NMAX;
    }

    std::vector<std::string> solve(bool perform_sim_annealing)
    {
        const int starting_depth = 5;
        const auto starting_states = get_states_at_depth(starting_depth);

        omp_set_num_threads(12);
        #pragma omp parallel for schedule(dynamic, 1)
        for (int th_index = 0; th_index < starting_states.size(); ++th_index)
        {
            simulate(starting_states[th_index], starting_depth, NMAX);
        }

        std::vector<std::string> result(data.universally_liked.begin(), data.universally_liked.end());

        if (perform_sim_annealing)
        {
            HybridSimulatedAnnealing hsa_solver(data);
            best_simulation_state = hsa_solver.attempt_improvement(best_simulation_state);
        }
        std::cout << "Score = " << data.nr_clients - best_simulation_state.clients_lost << "\n\n";

        for (int ingredient_index = 0; ingredient_index < data.ingredients.size(); ++ingredient_index)
            if ((*best_simulation_state.is_ingredient_chosen)[ingredient_index])
                result.push_back(data.ingredients[ingredient_index]);
        return result;
    }
};

int main()
{
    const std::string in_prefix = "../../input_files/";
    const std::string out_prefix = "../../output_files/sol2/";
    const std::array<std::string, 5> input_files = {"a_an_example.in", "b_basic.in", "c_coarse.in",
                                                    "d_difficult.in", "e_elaborate.in"};

    for (const auto &input_file: input_files)
    {
        Data data(in_prefix + input_file);
        std::cout << "Successfully read " << data.ingredients.size() << " unique ingredients and " << data.nr_clients
                  << " clients .\n";

        Solver solver(data);
        bool perform_sim_annealing = std::string("de").find(input_file[0]) != std::string::npos;
        const auto result = solver.solve(perform_sim_annealing);

        const auto out_filename = out_prefix + input_file.substr(0, (input_file.find('.'))) + ".out";
        Data::write_to_file(out_filename, result);
    }
    return 0;
}