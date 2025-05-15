#include <map>
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
        return elapsed.count() >= 30;
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
                    std::cout << "New best state has " << best_simulation_state.clients_lost << " clients lost\n";
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

    std::vector<int> get_client_satisfaction()
    {
        std::vector<int> client_satisfaction(data.client_to_satisfaction_req.begin(), data.client_to_satisfaction_req.end());
        for (int ingredient_index = 0; ingredient_index < data.ingredients.size(); ++ingredient_index)
        {
            const auto &ingredient = data.ingredients[ingredient_index];
            const auto ingr_fans_it = data.ingr_to_fans.find(ingredient);
            const auto ingr_haters_it = data.ingr_to_haters.find(ingredient);

            if ((*best_simulation_state.is_ingredient_chosen)[ingredient_index])
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

    int get_nr_clients_won_by_removal(const std::vector<int> &client_satisfaction, int ingredient_index)
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

    void apply_removal(std::vector<int> &client_satisfaction, int ingredient_index)
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

    int get_nr_clients_won_by_addition(const std::vector<int> &client_satisfaction, int ingredient_index)
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

    void apply_addition(std::vector<int> &client_satisfaction, int ingredient_index)
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

    void perform_local_search()
    {
        // Get the client satisfaction for the best solution found by the simulation
        auto client_satisfaction = get_client_satisfaction();
        int clients_gained = 0;

        bool improved = true;
        while (improved)
        {
            improved = false;

            for (int ingredient_index = 0; ingredient_index < data.ingredients.size(); ++ingredient_index)
                if ((*best_simulation_state.is_ingredient_chosen)[ingredient_index])
                {
                    const int removal_gain = get_nr_clients_won_by_removal(client_satisfaction, ingredient_index);
                    if (removal_gain > 0)
                    {
                        (*best_simulation_state.is_ingredient_chosen)[ingredient_index] = false;
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
                        (*best_simulation_state.is_ingredient_chosen)[ingredient_index] = true;
                        apply_addition(client_satisfaction, ingredient_index);

                        improved = true;
                        clients_gained += addition_gain;
                    }
                }
        }
        std::cout << "Gained " << clients_gained << " clients after local search\n";
    }

public:
    explicit Solver(const Data &data)
            : data(data), start(std::chrono::steady_clock::now())
    {
        best_simulation_state.clients_lost = NMAX;
    }

    std::vector<std::string> solve()
    {
        const int starting_depth = 5;
        const auto starting_states = get_states_at_depth(starting_depth);

        omp_set_num_threads(12);
        #pragma omp parallel for schedule(dynamic, 1)
        for (int th_index = 0; th_index < starting_states.size(); ++th_index)
        {
            simulate(starting_states[th_index], starting_depth, NMAX);
        }

        std::cout << "Score = " << data.nr_clients - best_simulation_state.clients_lost << "\n\n";
        std::vector<std::string> result(data.universally_liked.begin(), data.universally_liked.end());

        perform_local_search();

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

    for (const auto &input_file: input_files)
    {
        Data data(in_prefix + input_file);
        std::cout << "Successfully read " << data.ingredients.size() << " unique ingredients and " << data.nr_clients
                  << " clients .\n";

        Solver solver(data);
        const auto result = solver.solve();

        const auto out_filename = out_prefix + input_file.substr(0, (input_file.find('.'))) + ".out";
        Data::write_to_file(out_filename, result);
    }
    return 0;
}