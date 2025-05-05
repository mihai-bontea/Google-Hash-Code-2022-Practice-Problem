#pragma once
#include <fstream>
#include <vector>
#include <unordered_map>

struct Data
{
    int nr_clients;
    std::unordered_map<std::string, std::vector<int>> ingr_to_fans, ingr_to_haters;
    std::vector<std::string> ingredients;

    Data(const std::string& filename)
    {
        std::ifstream fin(filename);
        fin >> nr_clients;

        for (int client_id = 0; client_id < nr_clients; ++client_id)
        {
            int ingr_liked, ingr_disliked;
            std::string ingredient;

            auto ingredient_is_new = [&](const std::string& ingredient){
                const auto ingr_it1 = ingr_to_fans.find(ingredient);
                const auto ingr_it2 = ingr_to_haters.find(ingredient);
                return (ingr_it1 == ingr_to_fans.end() && ingr_it2 == ingr_to_haters.end());
            };

            fin >> ingr_liked;
            while (ingr_liked--)
            {
                fin >> ingredient;

                if (ingredient_is_new(ingredient))
                    ingredients.push_back(ingredient);

                ingr_to_fans[ingredient].push_back(client_id);
            }
            fin >> ingr_disliked;
            while(ingr_disliked--)
            {
                fin >> ingredient;
                if (ingredient_is_new(ingredient))
                    ingredients.push_back(ingredient);

                ingr_to_haters[ingredient].push_back(client_id);
            }
        }
    }

    static void write_to_file(const std::string& filename, const std::vector<std::string>& solution)
    {
        std::ofstream fout(filename);
        fout << solution.size() << " ";
        for (const auto& ingredient : solution)
            fout << ingredient << " ";
    }
};