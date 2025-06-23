#pragma once

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cassert>
#include <unordered_map>

class GpuIngredientMap {
public:
    int* ingr_to_fans = nullptr;
    int* ingr_to_haters = nullptr;

    void upload(const std::unordered_map<std::string, std::vector<int>>& fans,
        const std::unordered_map<std::string, std::vector<int>>& haters)
    {
        //        allocateAndCopy(ingr_to_fans, fans);
        //        allocateAndCopy(ingr_to_haters, haters);
        //        cudaMalloc(&ingr_to_fans, sizeof(int) * 10000 * 60);
        //        cudaMalloc(&ingr_to_haters, sizeof(int) * 10000 * 60);


    }

    void allocate_and_copy_map(int*& flattened_ptr, const std::unordered_map<std::string, std::vector<int>>& ingredient_to_clients)
    {
        //        cudaMalloc(&flattened_ptr, sizeof(int) * 10000 * 60);

        std::vector<int> temp(10000 * 60, 0);
        for (const auto& [ingredient, clients] : ingredient_to_clients)
        {
            auto it = std::find(ingredients.begin(), ingredients.end(), ingredient);
            if (it != ingredients.end()) {
                int index = std::distance(ingredients.begin(), it);

                temp[index * 60] = clients.size();
                int pos = 1;
                for (int client_id : clients)
                {
                    temp[index * 60 + pos++] = client_id;
                }
            }

        }
        cudaMemcpy(flattened_ptr, temp.data(), sizeof(int) * temp.size(), cudaMemcpyHostToDevice);
    }

    ~GpuIngredientMap() {
        cleanup();
    }

private:
    template <typename T>
    void allocateAndCopy(T*& device_ptr, const std::vector<T>& host_vec) {
        cudaMalloc(&device_ptr, sizeof(T) * host_vec.size());
        cudaMemcpy(device_ptr, host_vec.data(), sizeof(T) * host_vec.size(), cudaMemcpyHostToDevice);
    }

    void cleanup()
    {
        //        cudaFree(d_ingr_to_fans);
        //        cudaFree(d_ingr_to_fans_offsets);
        //        cudaFree(d_ingr_to_haters);
        //        cudaFree(d_ingr_to_haters_offsets);
    }
};

