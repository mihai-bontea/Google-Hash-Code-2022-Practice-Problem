#pragma once

#include <cuda_runtime.h>
#include "Individual.h"
#include "Defines.h"


class IFitnessEvaluator
{
protected:
    const Data& data;
    std::array<Individual, POPULATION_SIZE>& population;

public:
    virtual ~IFitnessEvaluator() = default;

    IFitnessEvaluator(const Data& data, std::array<Individual, POPULATION_SIZE>& population)
    : data(data)
    , population(population)
    {}

    virtual void evaluate() = 0;
};


class CpuFitnessEvaluator : public IFitnessEvaluator
{
public:
    using IFitnessEvaluator::IFitnessEvaluator;

    void evaluate() override
    {

    }
};


__global__ static void evaluate_satisfaction_kernel(
    const int num_ingredients,
    const int num_clients,
    const uint8_t* ingr_chosen, // NUM_INDIVIDUALS x NUM_INGREDIENTS
    const int* ingr_to_fans,   // NUM_INGREDIENTS × MAX_INGR_RELATIONS
    const int* ingr_to_haters, // NUM_INGREDIENTS × MAX_INGR_RELATIONS
    const int* client_to_satisfaction_req,
    int* client_satisfaction,
    int* fitness_scores) // Output: NUM_INDIVIDUALS)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= POPULATION_SIZE) return;

    const uint8_t* ingr_chosen_ = &ingr_chosen[tid]; // ???
    //int* output = &client_satisfaction[tid * num_clients]; // each thread writes to its own output

    //// Init satisfaction to 0
    //for (int i = 0; i < num_clients; ++i)
    //    output[i] = 0;

    //for (int ingredient_index = 0; ingredient_index < num_ingredients; ++ingredient_index)
    //{
    //    bool chosen = (bitset[ingredient_index / 32] >> (ingredient_index % 32)) & 1;

    //    const int* fans = &ingr_to_fans[ingredient_index * MAX_INGR_RELATIONS];
    //    const int* haters = &ingr_to_haters[ingredient_index * MAX_INGR_RELATIONS];

    //    int fan_count = fans[0];
    //    int hater_count = haters[0];

    //    if (chosen)
    //    {
    //        // Penalize haters
    //        for (int i = 0; i < hater_count; ++i) {
    //            int client_id = haters[i + 1];
    //            if (client_id >= 0 && client_id < num_clients)
    //                output[client_id]--;
    //        }
    //    }
    //    else
    //    {
    //        // Penalize fans
    //        for (int i = 0; i < fan_count; ++i) {
    //            int client_id = fans[i + 1];
    //            if (client_id >= 0 && client_id < num_clients)
    //                output[client_id]--;
    //        }
    //    }
    //}
}

class GpuFitnessEvaluator : public IFitnessEvaluator
{
private:
    uint8_t* ingr_chosen;
    int* fitness_scores;
    int* client_satisfaction;

    std::vector<uint8_t> get_flattened_bitsets()
    {
        std::vector<uint8_t> flat;
        flat.reserve(POPULATION_SIZE * MAX_INGREDIENTS);
        for (const auto& individual : population)
        {
            for (size_t i = 0; i < MAX_INGREDIENTS; ++i)
            {
                flat.push_back((*individual.genes)[i] ? 1 : 0);
            }
        }
        return flat;
    }

public:
    GpuFitnessEvaluator(const Data& data, std::array<Individual, POPULATION_SIZE>& population)
        : IFitnessEvaluator(data, population)
        , ingr_chosen(nullptr)
        , fitness_scores(nullptr)
        , client_satisfaction(nullptr)
    {
        cudaMalloc(&ingr_chosen, sizeof(uint8_t) * POPULATION_SIZE * MAX_INGREDIENTS);
        cudaMalloc(&fitness_scores, sizeof(int) * POPULATION_SIZE);
        cudaMalloc(&client_satisfaction, sizeof(int) * POPULATION_SIZE * MAX_CLIENTS);
    }

    ~GpuFitnessEvaluator()
    {
        cudaFree(ingr_chosen);
        cudaFree(fitness_scores);
    }

    void evaluate() override
    {
        const auto flattened_bitsets = get_flattened_bitsets();
        cudaMemcpy(ingr_chosen, flattened_bitsets.data(), flattened_bitsets.size(), cudaMemcpyHostToDevice);
           
        // Call the kernel
        const int threadsPerBlock = 256;
        const int numBlocks = (POPULATION_SIZE + threadsPerBlock - 1) / threadsPerBlock;

        evaluate_satisfaction_kernel << <numBlocks, threadsPerBlock >> > (
            data.nr_ingredients,
            data.nr_clients,
            ingr_chosen,
            data.gpu_ingredient_map.ingr_to_fans,
            data.gpu_ingredient_map.ingr_to_haters,
            data.gpu_ingredient_map.client_to_satisfaction_req,
            client_satisfaction,
            fitness_scores);
    }
};
