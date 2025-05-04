#include <array>
#include <iostream>
#include <bitset>
#include <set>
#define MAX_CLIENTS 100000

#include "Data.h"

struct SimulationState
{
    int simulation_id;
    std::bitset<MAX_CLIENTS> is_client_lost;
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

        std::set<std::string> unique_ingredients;
        for (const auto& [ingr, _] : data.ingr_to_fans)
            unique_ingredients.insert(ingr);
        for (const auto& [ingr, _] : data.ingr_to_haters)
            unique_ingredients.insert(ingr);

        std::cout << "Successfully read " << unique_ingredients.size() << " unique ingredients.\n\n";
    }
    return 0;
}
