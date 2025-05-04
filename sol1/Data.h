#pragma once
#include <fstream>
#include <vector>
#include <unordered_map>

struct Data
{
    int nr_clients;
    std::unordered_map<std::string, std::vector<int>> ingr_to_fans, ingr_to_haters;

    Data(const std::string& filename)
    {
        std::ifstream fin(filename);
        fin >> nr_clients;

        for (int client_id = 0; client_id < nr_clients; ++client_id)
        {
            int ingr_liked, ingr_disliked;
            std::string ingredient;

            fin >> ingr_liked;
            while (ingr_liked--)
            {
                fin >> ingredient;
                ingr_to_fans[ingredient].push_back(client_id);
            }
            fin >> ingr_disliked;
            while(ingr_disliked--)
            {
                fin >> ingredient;
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