#include <array>
#include <iostream>
#include <bitset>
#define MAX_CLIENTS 100000
#define MAX_INGREDIENTS 10000
#define NMAX 99999999

#include "Data.h"

struct SimulationState
{
    int depth, clients_lost;
    std::bitset<MAX_CLIENTS> is_client_remaining;
    std::bitset<MAX_INGREDIENTS> is_ingredient_chosen;

    SimulationState(): depth(0), clients_lost(0)
    {
        is_client_remaining.set();
        is_ingredient_chosen.reset();
    }
};

class Solver
{
private:
    const Data& data;
    SimulationState best_simulation_state;

    SimulationState update_for_ingr_addition(const SimulationState& simulation_state, int ingredient_index)
    {
        SimulationState updated_state = simulation_state;
        updated_state.depth++;
        updated_state.is_ingredient_chosen[ingredient_index] = true;

        const auto& ingredient = data.ingredients[ingredient_index];
        const auto clients_it = data.ingr_to_haters.find(ingredient);

        // Nobody dislikes this ingredient
        if (clients_it == data.ingr_to_haters.end())
            return updated_state;

        // All clients who dislike this ingredient are lost
        for (int client_index : clients_it->second)
        {
            if (updated_state.is_client_remaining[client_index])
            {
                updated_state.is_client_remaining[client_index] = false;
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
            if (updated_state.is_client_remaining[client_index])
            {
                updated_state.is_client_remaining[client_index] = false;
                updated_state.clients_lost++;
            }
        }

        return updated_state;
    }

    void simulate(SimulationState simulation_state, int ingredient_index)
    {
        if (ingredient_index == data.ingredients.size())
        {
            if (simulation_state.clients_lost < best_simulation_state.clients_lost)
            {
                best_simulation_state = simulation_state;
            }
            return;
        }

        const double ingr_appreciation = (double)data.ingr_to_fans.size() / (double)data.ingr_to_haters.size();
        if (ingr_appreciation > 0)
        {
            simulate(update_for_ingr_addition(simulation_state, ingredient_index), ingredient_index + 1);
            simulate(update_for_ingr_removal(simulation_state, ingredient_index), ingredient_index + 1);
        }
        else
        {
            simulate(update_for_ingr_removal(simulation_state, ingredient_index), ingredient_index + 1);
            simulate(update_for_ingr_addition(simulation_state, ingredient_index), ingredient_index + 1);
        }
    }
public:
    explicit Solver(const Data& data): data(data)
    {
        best_simulation_state.clients_lost = NMAX;
    }

    std::vector<std::string> solve()
    {
        simulate(SimulationState(), 0);
        std::cout << "Score = " << data.nr_clients - best_simulation_state.clients_lost << "\n\n";

        std::vector<std::string> result;

        for (int ingredient_index = 0; ingredient_index < data.ingredients.size(); ++ingredient_index)
            if (best_simulation_state.is_ingredient_chosen[ingredient_index])
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
        std::cout << "Successfully read " << data.ingredients.size() << " unique ingredients.\n";

        Solver solver(data);
        auto result = solver.solve();

        const auto out_filename = out_prefix + input_file.substr(0, (input_file.find('.'))) + ".out";
        Data::write_to_file(out_filename, result);

        if (input_file == "c_coarse.in")
            break;
    }
    return 0;
}
