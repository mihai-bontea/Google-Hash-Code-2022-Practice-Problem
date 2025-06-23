#pragma once

#include <cuda_runtime.h>
#include "Individual.h"
#include "Defines.h"


class IFitnessEvaluator
{
protected:
    std::array<Individual, POPULATION_SIZE>& population;

public:
    virtual ~IFitnessEvaluator() = default;

    IFitnessEvaluator(std::array<Individual, POPULATION_SIZE>& population)
        : population(population)
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
    const int num_individuals,
    const int num_ingredients,
    const int num_clients,
    const bool* ingr_chosen, // NUM_INDIVIDUALS x NUM_INGREDIENTS
    const int* ingr_to_fans,   // NUM_INGREDIENTS × MAX_INGR_RELATIONS
    const int* ingr_to_haters, // NUM_INGREDIENTS × MAX_INGR_RELATIONS
    const int* client_to_satisfaction_req,
    int* fitness_scores) // Output: NUM_INDIVIDUALS)
{
    //int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //if (tid >= num_individuals) return;

    //const uint32_t* bitset = &bitsets[tid]; // points to bit pattern of this individual
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
public:
    void evaluate() override
    {
        const auto flattened_bitsets = get_flattened_bitsets();
    }
private:
    

    std::vector<bool> get_flattened_bitsets()
    {
        std::vector<bool> flat;
        flat.reserve(POPULATION_SIZE * MAX_INGREDIENTS);
        for (const auto& individual : population)
        {
            for (size_t i = 0; i < MAX_INGREDIENTS; ++i)
            {
                flat.push_back((*individual.genes)[i]);
            }
        }
        return flat;
    }
};



//void evalute_fitness(const std::vector<bool>& flattened_bitsets)
//{
//    int threads = 128;
//    // num_individals = 100
//    int blocks = (100 + threads - 1) / threads;
//
//    evaluate_satisfaction_kernel << <blocks, threads >> > (
//        device_bitsets,
//        device_client_satisfaction,
//        NUM_INDIVIDUALS,
//        NUM_INGREDIENTS,
//        NUM_CLIENTS,
//        device_ingr_to_fans,
//        device_ingr_to_haters,
//        MAX_INGR_RELATIONS
//        );
//}