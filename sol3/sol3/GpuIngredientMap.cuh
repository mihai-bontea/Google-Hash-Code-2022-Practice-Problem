#pragma once

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cassert>
#include <unordered_map>

#include "Defines.h"

class GpuIngredientMap
{
public:
    int* ingr_to_fans = nullptr;
    int* ingr_to_haters = nullptr;

    void upload(
        const std::unordered_map<std::string, std::vector<int>>& fans,
        const std::unordered_map<std::string, std::vector<int>>& haters,
        const std::vector<std::string>& ingredients)
    {
                //allocateAndCopy(ingr_to_fans, fans);
                //allocateAndCopy(ingr_to_haters, haters);
        //        cudaMalloc(&ingr_to_fans, sizeof(int) * 10000 * 60);
        //        cudaMalloc(&ingr_to_haters, sizeof(int) * 10000 * 60);
        allocate_and_copy_map(ingr_to_fans, fans, ingredients);
        allocate_and_copy_map(ingr_to_haters, haters, ingredients);
    }

    void allocate_and_copy_map(
        int*& flattened_ptr,
        const std::unordered_map<std::string, std::vector<int>>& ingredient_to_clients,
        const std::vector<std::string>& ingredients)
    {
        cudaMalloc(&flattened_ptr, sizeof(int) * MAX_INGREDIENTS * MAX_INGR_RELATIONS);

        std::vector<int> temp(MAX_INGREDIENTS * MAX_INGR_RELATIONS, 0);
        for (const auto& [ingredient, clients] : ingredient_to_clients)
        {
            auto it = std::find(ingredients.begin(), ingredients.end(), ingredient);
            if (it != ingredients.end())
            {
                int ingredient_index = std::distance(ingredients.begin(), it);

                temp[ingredient_index * MAX_INGR_RELATIONS] = clients.size();
                int next_pos = 1;
                for (int client_id : clients)
                {
                    temp[ingredient_index * MAX_INGR_RELATIONS + next_pos++] = client_id;
                }
            }

        }
        cudaMemcpy(flattened_ptr, temp.data(), sizeof(int) * temp.size(), cudaMemcpyHostToDevice);
    }

    ~GpuIngredientMap()
    {
        cleanup();
    }

private:
    template <typename T>
    void allocateAndCopy(T*& device_ptr, const std::vector<T>& host_vec)
    {
        cudaMalloc(&device_ptr, sizeof(T) * host_vec.size());
        cudaMemcpy(device_ptr, host_vec.data(), sizeof(T) * host_vec.size(), cudaMemcpyHostToDevice);
    }

    void cleanup()
    {
        cudaFree(ingr_to_fans);
        cudaFree(ingr_to_haters);
    }
};

