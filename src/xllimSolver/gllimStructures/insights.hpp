#ifndef INSIGHTS_HPP
#define INSIGHTS_HPP

#include <armadillo>

using namespace arma;

struct InitialisationInsights
{
    std::chrono::seconds time;
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    unsigned N_obs;
    unsigned gllim_em_iteration;
    double gllim_em_floor;
    unsigned gmm_kmeans_iteration;
    unsigned gmm_em_iteration;
    double gmm_floor;
    unsigned nb_experiences;

    InitialisationInsights() : time(std::chrono::seconds(0)), N_obs(0)  {}
    InitialisationInsights(std::chrono::seconds time, std::chrono::high_resolution_clock::time_point start_time, std::chrono::high_resolution_clock::time_point end_time, unsigned N_obs, unsigned gllim_em_iteration, double gllim_em_floor, unsigned gmm_kmeans_iteration, unsigned gmm_em_iteration, double gmm_floor, unsigned nb_experiences)
        : time(time), start_time(start_time), end_time(end_time), N_obs(N_obs), gllim_em_iteration(gllim_em_iteration), gllim_em_floor(gllim_em_floor), gmm_kmeans_iteration(gmm_kmeans_iteration), gmm_em_iteration(gmm_em_iteration), gmm_floor(gmm_floor), nb_experiences(nb_experiences) {}
};

struct TrainingInsights
{
    std::chrono::seconds time;
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    unsigned N_obs;
    unsigned max_iteration;
    double ratio_ll;
    double floor;

    TrainingInsights() : time(std::chrono::seconds(0)), N_obs(0) {}
    TrainingInsights(std::chrono::seconds time, std::chrono::high_resolution_clock::time_point start_time, std::chrono::high_resolution_clock::time_point end_time, unsigned N_obs, unsigned max_iteration, double ratio_ll, double floor)
        : time(time), start_time(start_time), end_time(end_time), N_obs(N_obs), max_iteration(max_iteration), ratio_ll(ratio_ll), floor(floor) {}
};

struct Insights
{
    std::chrono::seconds time;
    vec log_likelihood;
    InitialisationInsights initialisation;
    TrainingInsights training;

    Insights() : time(std::chrono::seconds(0)) {}
};

#endif // INSIGHTS_HPP
