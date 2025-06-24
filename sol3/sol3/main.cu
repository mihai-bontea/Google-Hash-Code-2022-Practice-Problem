#include <array>
#include <iostream>

#include "Data.h"
#include "GeneticSolver.h"

int main()
{
    const std::string in_prefix = "../../input_files/";
    const std::string out_prefix = "../../output_files/sol3/";
    const std::array<std::string, 5> input_files = { "a_an_example.in", "b_basic.in", "c_coarse.in",
                                                    "d_difficult.in", "e_elaborate.in" };

    for (const auto& input_file : input_files)
    {
        if (input_file != "e_elaborate.in")
            continue;

        Data data(in_prefix + input_file);
        std::cout << "Successfully read " << data.ingredients.size() << " unique ingredients and " << data.nr_clients
            << " clients .\n";

        GeneticSolver solver(data);
        const auto result = solver.solve();

        const auto out_filename = out_prefix + input_file.substr(0, (input_file.find('.'))) + ".out";
        Data::write_to_file(out_filename, result);
    }
    return 0;
}