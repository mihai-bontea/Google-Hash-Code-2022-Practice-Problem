#pragma once

#include <omp.h>
#include <cuda_runtime.h>

#include "Defines.h"
#include "Individual.h"

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
private:
    int get_clients_lost(const Individual& individual) const
    {
        int clients_lost = 0;

        std::bitset<MAX_CLIENTS> is_client_remaining;
        is_client_remaining.set();

        const auto& genes = *individual.genes;
        for (int ingredient_index = 0; ingredient_index < data.nr_ingredients; ++ingredient_index)
        {
            const auto& ingredient = data.ingredients[ingredient_index];
            const auto ingr_fans_it = data.ingr_to_fans.find(ingredient);
            const auto ingr_haters_it = data.ingr_to_haters.find(ingredient);

            if (genes[ingredient_index])
            {
                // Decrease satisfaction for clients who dislike this ingredient
                if (ingr_haters_it != data.ingr_to_haters.end())
                    for (int client_id : ingr_haters_it->second)
                    {
                        if (is_client_remaining[client_id])
                            clients_lost++;
                        is_client_remaining[client_id] = false;
                    }
            }
            else
            {
                // Decrease satisfaction for clients who like this ingredient
                if (ingr_fans_it != data.ingr_to_fans.end())
                    for (int client_id : ingr_fans_it->second)
                    {
                        if (is_client_remaining[client_id])
                            clients_lost++;
                        is_client_remaining[client_id] = false;
                    }
            }
        }
        return clients_lost;
    }

public:
    using IFitnessEvaluator::IFitnessEvaluator;

    void evaluate() override
    {
        #pragma omp parallel for num_threads(12)
        for (int individual_index = 0; individual_index < POPULATION_SIZE; ++individual_index)
        {
            const int clients_remaining = data.nr_clients - get_clients_lost(population[individual_index]);
            population[individual_index].fitness = clients_remaining;
        }
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
