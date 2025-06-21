#include <array>
#include <iostream>
#include "Data.h"

int main()
{
    const std::string in_prefix = "../../input_files/";
    const std::string out_prefix = "../../output_files/sol3/";
    const std::array<std::string, 5> input_files = { "a_an_example.in", "b_basic.in", "c_coarse.in",
                                                    "d_difficult.in", "e_elaborate.in" };

    for (const auto& input_file : input_files)
    {
        Data data(in_prefix + input_file);
        std::cout << "Successfully read " << data.ingredients.size() << " unique ingredients and " << data.nr_clients
            << " clients .\n";

    
    }
    return 0;
}