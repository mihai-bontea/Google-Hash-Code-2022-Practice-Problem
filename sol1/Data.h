#pragma once
#include <vector>
#include <fstream>
#include <unordered_map>

#define NMAX 99999999
#define MAX_CLIENTS 100000
#define MAX_INGREDIENTS 10000

struct Data
{
    int nr_clients;
    std::unordered_map<std::string, std::vector<int>> ingr_to_fans, ingr_to_haters;
    std::vector<std::string> ingredients, universally_liked;

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
        if (filename.contains("d_difficult") || filename.contains("e_elaborate"))
        {
            std::erase_if(ingredients, [&](auto& ingredient ) {
                if (ingr_to_haters.find(ingredient) == ingr_to_haters.end())
                {
                    universally_liked.push_back(ingredient);
                    return true;
                }
                return false;
            });
            std::cout << "Removed " << universally_liked.size() << " ingredients that everyone likes.\n";
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