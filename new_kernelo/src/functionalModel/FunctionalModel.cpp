#include "FunctionalModel.hpp"
#include "../dataGeneration/generator/GeneratorFactory.hpp"
#include "../dataGeneration/generator/Generator.hpp"
#include "../utils/utils.hpp"
#include <omp.h>

// TODO

std::tuple<mat, mat> FunctionalModel::genData(unsigned N, const std::string &generator_type, vec &covariance, unsigned seed)
{
    unsigned dimension_D = this->getDimensionY();
    unsigned dimension_L = this->getDimensionX();
    mat x_gen = mat(N, dimension_L);
    mat y_gen = mat(N, dimension_D);

    // generate X
    std::shared_ptr<DataGeneration::Generator> generator = DataGeneration::GeneratorFactory::create(generator_type, seed);
    generator->execute(x_gen);

    // create a vector of random values under a normal distribution with 0 mean and 1 variance
    std::normal_distribution<double> normal_distribution(0, 1);
    std::mt19937_64 engine;
    engine.seed(seed);

    // generate Y
    vec noise(dimension_D);
    vec y_temp(dimension_D);

#pragma omp parallel for
    for (unsigned i = 0; i < N; i++)
    {
        // calculate F(X)
        this->F(x_gen.row(i).t(), y_temp);

        // add noise
        for (unsigned j = 0; j < dimension_D; j++)
        {
            noise(j) = normal_distribution(engine);
            y_gen(i, j) = y_temp(j) + noise(j) * covariance(j);
        }
    }

    return std::tuple<mat, mat>(x_gen, y_gen);
}

std::tuple<mat, mat> FunctionalModel::genData(unsigned N, const std::string &generator_type, double noise_ratio, unsigned seed)
{
    unsigned dimension_D = this->getDimensionY();
    unsigned dimension_L = this->getDimensionX();
    mat x_gen = mat(N, dimension_L);
    mat y_gen = mat(N, dimension_D);

    // generate X
    std::shared_ptr<DataGeneration::Generator> generator = DataGeneration::GeneratorFactory::create(generator_type, seed);
    generator->execute(x_gen);

    // create a vector of random values under a normal distribution with 0 mean and 1 variance
    std::normal_distribution<double> normal_distribution(0, 1);
    std::mt19937_64 engine;
    engine.seed(seed);

    // generate Y
    vec noise(dimension_D);
    vec y_temp(dimension_D);

#pragma omp parallel for
    for (unsigned i = 0; i < N; i++)
    {
        // calculate F(X)
        this->F(x_gen.row(i).t(), y_temp);

        // add noise
        for (unsigned j = 0; j < dimension_D; j++)
        {
            noise(j) = normal_distribution(engine);
            y_gen(i, j) = y_temp(j) + noise(j) * y_temp(j) / noise_ratio;
        }
    }

    return std::tuple<mat, mat>(x_gen, y_gen);
}

vec FunctionalModel::targetDensity(const mat &x, const vec &y, const vec &y_err, const vec &covariance, bool log)
{
    vec densities(x.n_cols);
    vec F_on_x(y.n_rows);
    mat F_on_x_mat(y.n_rows,1); // TODO: redundant but compilation error otherwise
    cube  covariance_matrix(y.n_rows, y.n_rows, 1);
    covariance_matrix.slice(0) = diagmat(pow(y_err, 2) + pow(covariance, 2));
    gmm_full gmm;
    rowvec weight(1, fill::value(1));
    for (unsigned n = 0; n < x.n_cols; ++n)
    { // TODO/NOTE: vectorisation of physical models
        if (any(x.col(n) > 1 || x.col(n) < 0))
        {
            densities(n) = -datum::inf;
        }
        this->F(x.col(n), F_on_x);
        F_on_x_mat.col(0) = F_on_x;
        gmm.set_params(F_on_x_mat, covariance_matrix, weight);
        densities(n) = gmm.log_p(y);
        // densities(n) = utils::logDensity(y, F_on_x, covariance_matrix.slice(0));
    }
    return densities;
}

void FunctionalModel::targetDensity(vec &x, vec &y, vec &y_err, double noise_ratio, bool log)
{
}

vec FunctionalModel::propositionDensity(const mat &x, const vec &weight, const mat &mean, const cube &covariance, bool log)
{
    gmm_full gmm;
    gmm.set_params(mean, covariance, weight.t());
    return gmm.log_p(x).t();
    // return utils::logSumExp(utils::logDensity(x, weight.t(), mean, covariance), 1);
}

mat FunctionalModel::importanceSampling(std::vector<std::tuple<const vec, const mat, const cube>> proposition_gmms, const mat y, const mat y_err, const vec covariance, const unsigned N_0, const unsigned B, const unsigned J)
{
    // TODO : modifiy and adapt this code for IMIS
    // IDEE: les gmm pour chaque observations (list de (vec &weights, mat &means, cube &covariances)) sont attributs de gllim ? ou pas. Non on ne va pas enregistrer N_obs données
    // Parce que c'est peut être beaucoup d'avoir une liste de longueur N_obs constitué de weight, mean, cov pouvant être très grand aussi
    // bon tant pis. Au pire il faudrait couper les observations en plusieurs.
    // IDEE: si les propositions ne servent quà faire des gmm. On pourrait transformer le tuple en gmm dans le binding par example. Puis supprimer le tuple !! (attention aux fuites mémoire). Mais ce que j'ai fait c'est très bien aussi !
    // IDEE faire un namespace séparé avec les méthodes de importance sampling pour ne pas surcharger ce fichier ?

    // TODO faire un code global qui return une structure (struct) ImportanceSamplingResult. a l'instar de old Kernelo !
    // il faudra réfléchir à la sortie de gllim.predict() (inverseDensity + gmm/proposition). Vérifier qu'on a bien meanPredResult.mean = Somme de meanPredResult.gmm_weights * meanPredResult.gmm_means

    // mat ImportanceSampler::executeAll(std::vector<std::shared_ptr<ISProposition>> &isProposition_list, const mat &y_obs) {
    unsigned N_samples = N_0 + B * J;                     // for imis
    unsigned L = std::get<1>(proposition_gmms[0]).n_rows; // get the number of rows in the first GMM mean matrix
    unsigned N_obs = y.n_rows;
    mat results(L, N_obs);
    mat samples(L, N_samples);
    vec weights(N_samples);
    double sum_weights;
    double sum_weights_2;
    vec target_log_densities(N_samples);
    vec proposition_log_densities(N_samples);
    gmm_full proposition_gmm;

    // #pragma omp parallel for shared(results, y_cov, y_obs, isProposition_list,L,N_obs)// private(samples, weights, target_log_densities, proposition_log_densities) shared(results, y_cov, y_obs, isProposition_list,L,N_obs)
    // check branch benchmark_with_optimized_methods to get the working parallelized ImportanceSampling() method
    for (unsigned n_obs = 0; n_obs < N_obs; ++n_obs)
    {
        // ===================== void FunctionalModel::importanceSamplingCore(...) ========================
        proposition_gmm.set_params(std::get<1>(proposition_gmms[n_obs]), std::get<2>(proposition_gmms[n_obs]), std::get<0>(proposition_gmms[n_obs]).t());
        samples = proposition_gmm.generate(N_samples);                                                                                                                             // sample with GLLiM-GMM
        target_log_densities = targetDensity(samples, y.row(n_obs).t(), y_err.row(n_obs).t(), covariance);                                                                         // compute the target log probability density function (PDF)
        proposition_log_densities = propositionDensity(samples, std::get<0>(proposition_gmms[n_obs]), std::get<1>(proposition_gmms[n_obs]), std::get<2>(proposition_gmms[n_obs])); // compute the target log probability density function (PDF)
        weights = target_log_densities - proposition_log_densities;                                                                                                                // compute weights verifying numerical stability
        sum_weights = utils::logSumExp(weights);
        sum_weights_2 = utils::logSumExp(2 * weights);
        weights -= sum_weights;
        weights = exp(weights);
        // ======================================================================================================

        // ===================== void FunctionalModel::importanceSamplingDiagnostic(...) ========================
        // Compute samples mean
        for (unsigned n = 0; n < N_samples; n++)
        {
            results.col(n_obs) += samples.col(n) * weights(n); // Compute samples mean
            // result.variance += weights_samples(n) * pow(X_samples.col(n) - result.mean, 2); // Compute samples variance
        }
        // IS diagnostic
        // + des prints !! (logger)
        // diagnostic.nb_effective_sample = find_finite(target_log_densities).n_elem;
        // diagnostic.effective_sample_size = exp(2 * sum_weights - sum_weights_2);
        // diagnostic.qn = exp(weights.max() - sum_weights);
        // ======================================================================================================
    }

    return results;
}