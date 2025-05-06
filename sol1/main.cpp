#include <array>
#include <cmath>
#include <bitset>
#include <chrono>
#include <iostream>
#include <functional>

#include <omp.h>

#include "Data.h"

struct SimulationState
{
    int depth = 0, clients_lost = 0;
    std::unique_ptr<std::bitset<MAX_CLIENTS>> is_client_remaining;
    std::unique_ptr<std::bitset<MAX_INGREDIENTS>> is_ingredient_chosen;

    SimulationState()
        : is_client_remaining(std::make_unique<std::bitset<MAX_CLIENTS>>())
        , is_ingredient_chosen(std::make_unique<std::bitset<MAX_INGREDIENTS>>())
    {
        is_client_remaining->set();
        is_ingredient_chosen->reset();
    }

    SimulationState(const SimulationState& other)
        : depth(other.depth)
        , clients_lost(other.clients_lost)
        , is_client_remaining(std::make_unique<std::bitset<MAX_CLIENTS>>(*other.is_client_remaining))
        , is_ingredient_chosen(std::make_unique<std::bitset<MAX_INGREDIENTS>>(*other.is_ingredient_chosen))
    {}

    SimulationState& operator=(const SimulationState& other)
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

    SimulationState(SimulationState&&) = default;

    SimulationState& operator=(SimulationState&&) = default;
};

class Solver
{
private:
    const Data& data;
    SimulationState best_simulation_state;
    const std::chrono::steady_clock::time_point start;

    SimulationState update_for_ingr_addition(const SimulationState& simulation_state, int ingredient_index)
    {
        SimulationState updated_state = simulation_state;
        updated_state.depth++;
        (*updated_state.is_ingredient_chosen)[ingredient_index] = true;

        const auto& ingredient = data.ingredients[ingredient_index];
        const auto clients_it = data.ingr_to_haters.find(ingredient);

        // Nobody dislikes this ingredient
        if (clients_it == data.ingr_to_haters.end())
            return updated_state;

        // All clients who dislike this ingredient are lost
        for (int client_index : clients_it->second)
        {
            if ((*updated_state.is_client_remaining)[client_index])
            {
                (*updated_state.is_client_remaining)[client_index] = false;
                updated_state.clients_lost++;
            }
        }

        return updated_state;
    }

    SimulationState update_for_ingr_removal(const SimulationState& simulation_state, int ingredient_index)
    {
        SimulationState updated_state = simulation_state;
        updated_state.depth++;

        const auto& ingredient = data.ingredients[ingredient_index];
        const auto clients_it = data.ingr_to_fans.find(ingredient);

        // Nobody likes this ingredient
        if (clients_it == data.ingr_to_fans.end())
            return updated_state;

        // All clients who like this ingredient are lost
        for (int client_index : clients_it->second)
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
        return elapsed.count() >= 30;
    }

    std::vector<SimulationState> get_states_at_depth(const int max_depth)
    {
        // There are 2^(depth - 1) nodes(states)
        std::vector<SimulationState> states;
        states.reserve((size_t)pow(2, max_depth - 1));

        std::function<void(SimulationState)> simulate_to_depth = [&](SimulationState simulation_state)
        {
            if (simulation_state.depth == max_depth)
            {
                states.push_back(simulation_state);
            }
            else
            {
                simulate_to_depth(update_for_ingr_addition(simulation_state, simulation_state.depth));
                simulate_to_depth(update_for_ingr_removal(simulation_state, simulation_state.depth));
            }
        };
        simulate_to_depth(SimulationState());
        return states;
    }

    void simulate(SimulationState simulation_state, int ingredient_index)
    {
        if (ingredient_index == data.ingredients.size())
        {
            #pragma omp critical
            {
                if (simulation_state.clients_lost < best_simulation_state.clients_lost)
                {
                    best_simulation_state = simulation_state;
                    std::cout << "New best state has " << best_simulation_state.clients_lost << " clients lost\n";
                }
            }
            return;
        }

        bool prune = false;
        #pragma omp critical
        {
            // Prune branch if already worse than the best solution found so far
            if (simulation_state.clients_lost >= best_simulation_state.clients_lost)
                prune = true;
        }
        if (prune || is_timer_expired())
            return;

        const auto& ingredient = data.ingredients[ingredient_index];
        const auto ingr_fans_it = data.ingr_to_fans.find(ingredient);
        const auto ingr_haters_it = data.ingr_to_haters.find(ingredient);

        // No one dislikes this ingredient, no point checking for its removal
        if (ingr_haters_it == data.ingr_to_haters.end())
            simulate(update_for_ingr_addition(simulation_state, ingredient_index), ingredient_index + 1);
        // No one likes this ingredient, no point checking for its addition
        else if (ingr_fans_it == data.ingr_to_fans.end())
            simulate(update_for_ingr_removal(simulation_state, ingredient_index), ingredient_index + 1);
        // More people like this ingredient than dislike it, prioritize checking its addition
        else if (ingr_fans_it->second.size() > ingr_haters_it->second.size())
        {
            simulate(update_for_ingr_addition(simulation_state, ingredient_index), ingredient_index + 1);
            simulate(update_for_ingr_removal(simulation_state, ingredient_index), ingredient_index + 1);
        }
        // More people dislike this ingredient than ike it, prioritize checking its removal
        else
        {
            simulate(update_for_ingr_removal(simulation_state, ingredient_index), ingredient_index + 1);
            simulate(update_for_ingr_addition(simulation_state, ingredient_index), ingredient_index + 1);
        }
    }
public:
    explicit Solver(const Data& data)
        : data(data)
        , start(std::chrono::steady_clock::now())
    {
        best_simulation_state.clients_lost = NMAX;
    }

    std::vector<std::string> solve()
    {
        const int starting_depth = 3;
        const auto starting_states = get_states_at_depth(starting_depth);

        omp_set_num_threads((int)starting_states.size());
        #pragma omp parallel for
        for (int th_index = 0; th_index < starting_states.size(); ++th_index)
        {
            simulate(starting_states[th_index], starting_depth);
        }

        std::cout << "Score = " << data.nr_clients - best_simulation_state.clients_lost << "\n\n";
        std::vector<std::string> result;

        for (int ingredient_index = 0; ingredient_index < data.ingredients.size(); ++ingredient_index)
            if ((*best_simulation_state.is_ingredient_chosen)[ingredient_index])
                result.push_back(data.ingredients[ingredient_index]);
        return result;
    }
};

int main()
{
    const std::string in_prefix = "../../input_files/";
    const std::string out_prefix = "../../output_files/sol1/";
    const std::array<std::string, 5> input_files = {"a_an_example.in", "b_basic.in", "c_coarse.in",
                                                    "d_difficult.in", "e_elaborate.in"};

    for (const auto& input_file : input_files)
    {
        Data data(in_prefix + input_file);
        std::cout << "Successfully read " << data.ingredients.size() << " unique ingredients and " << data.nr_clients << " clients .\n";

        Solver solver(data);
        auto result = solver.solve();

        const auto out_filename = out_prefix + input_file.substr(0, (input_file.find('.'))) + ".out";
        Data::write_to_file(out_filename, result);
    }
    return 0;
}
