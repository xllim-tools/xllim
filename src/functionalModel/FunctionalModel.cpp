#include "FunctionalModel.hpp"
#include "../generator/GeneratorFactory.hpp"
// #include "../dataGeneration/generator/Generator.hpp"
#include "../utils/utils.hpp"
#include "../logging/logger.hpp"
// #include <omp.h>

// TODO

std::tuple<mat, mat> FunctionalModel::genData(unsigned N, const std::string &generator_type, vec &covariance, unsigned seed)
{
    unsigned dimension_D = getDimensionY();
    unsigned dimension_L = getDimensionX();
    mat x_gen = mat(dimension_L, N);
    mat y_gen = mat(dimension_D, N);

    // generate X
    std::shared_ptr<DataGeneration::Generator> generator = createGenerator(generator_type, seed);
    generator->execute(x_gen);

    // create a vector of random values under a normal distribution with 0 mean and 1 variance
    std::normal_distribution<double> normal_distribution(0, 1);
    std::mt19937_64 engine;
    engine.seed(seed);

    // #pragma omp parallel for
    for (unsigned j = 0; j < N; j++)
    {
        double noise;
        vec y_temp(dimension_D);

        // calculate F(X)
        F(x_gen.col(j), y_temp);

        // add noise
        for (unsigned i = 0; i < dimension_D; i++)
        {
            noise = normal_distribution(engine);
            y_gen(i, j) = y_temp(i) + noise * covariance(i);
        }
    }

    return std::tuple<mat, mat>(x_gen, y_gen);
}

std::tuple<mat, mat> FunctionalModel::genData(unsigned N, const std::string &generator_type, double noise_ratio, unsigned seed)
{
    unsigned dimension_D = getDimensionY();
    unsigned dimension_L = getDimensionX();
    mat x_gen = mat(dimension_L, N);
    mat y_gen = mat(dimension_D, N);

    // generate X
    std::shared_ptr<DataGeneration::Generator> generator = createGenerator(generator_type, seed);
    generator->execute(x_gen);

    // create a vector of random values under a normal distribution with 0 mean and 1 variance
    std::normal_distribution<double> normal_distribution(0, 1);
    std::mt19937_64 engine;
    engine.seed(seed);

    // #pragma omp parallel for
    for (unsigned j = 0; j < N; j++)
    {
        double noise;
        vec y_temp(dimension_D);

        // calculate F(X)
        F(x_gen.col(j), y_temp);

        // add noise
        for (unsigned i = 0; i < dimension_D; i++)
        {
            noise = normal_distribution(engine);
            y_gen(i, j) = y_temp(i) + noise * y_temp(i) * noise_ratio / 100.;
        }
    }

    return std::tuple<mat, mat>(x_gen, y_gen);
}

vec FunctionalModel::targetDensity(const mat &x, const vec &y, const vec &y_err, const vec &covariance, bool log)
{
    vec densities(x.n_cols);
    vec F_on_x(y.n_rows);
    mat F_on_x_mat(y.n_rows, 1); // TODO: redundant but compilation error otherwise
    mat covariance_matrix(y.n_rows, 1);
    covariance_matrix.col(0) = pow(y_err, 2) + pow(covariance, 2);
    gmm_diag gmm;
    rowvec weight(1, fill::value(1));
    for (unsigned n = 0; n < x.n_cols; ++n)
    {                                          // TODO/NOTE: vectorisation of physical models
        if (any(x.col(n) > 1 || x.col(n) < 0)) // if x is not in [0,1], F(x) may return nan
        {
            densities(n) = -datum::inf;
            // vec dummy(x.n_rows, fill::value(0.5)); // Force the sampling if x not in [0,1] by applying F on dummy median value
            // F(dummy, F_on_x);
        }
        else
        {
            F(x.col(n), F_on_x);
            F_on_x_mat.col(0) = F_on_x;
            gmm.set_params(F_on_x_mat, covariance_matrix, weight);
            densities(n) = gmm.log_p(y);
            // densities(n) = utils::logDensity(y, F_on_x, covariance_matrix.col(0)); // provides same results as gmm.log_p(y). I verified.
        }
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

ImportanceSamplingResult FunctionalModel::importanceSampling(const std::vector<std::tuple<vec, mat, cube>> &proposition_gmms, const mat y, const mat y_err, const unsigned N_0, const unsigned B, const unsigned J, const vec covariance, int idx_gaussian, int verbose, unsigned seed)
{
    const unsigned K = std::get<0>(proposition_gmms[0]).n_rows;

    if (idx_gaussian == -1) // apply IMIS on all gaussians
    {
        return importanceSamplingParsedGMMCheck(proposition_gmms, y, y_err, N_0, B, J, covariance, verbose, seed);
    }
    else if (idx_gaussian >= 0 && idx_gaussian < K) // apply IMIS on specific gaussian
    {
        const unsigned N_obs = proposition_gmms.size();
        const unsigned D = std::get<2>(proposition_gmms[0]).n_cols;
        cube covs_cube(D, D, 1);
        std::vector<std::tuple<vec, mat, cube>> proposition_gmm_one_gaussian;

        for (size_t n = 0; n < N_obs; n++)
        {
            covs_cube.slice(0) = std::get<2>(proposition_gmms[n]).slice(idx_gaussian);
            proposition_gmm_one_gaussian.push_back(std::make_tuple(
                vec(1, fill::value(1)),
                mat(std::get<1>(proposition_gmms[n]).col(idx_gaussian)),
                covs_cube));
        }
        return importanceSamplingParsedGMMCheck(proposition_gmm_one_gaussian, y, y_err, N_0, B, J, covariance, verbose, seed);
    }
    else
    {
        throw std::invalid_argument("idx_gaussian is out of range");
    }
}

ImportanceSamplingResult FunctionalModel::importanceSampling(const FullGMMResult fullGMM, const mat y, const mat y_err, const unsigned N_0, const unsigned B, const unsigned J, const vec covariance, int idx_gaussian, int verbose, unsigned seed)
{
    // retrieve the gmm parameters from de GLLiM prediction results. It corresponds to the proposition law for the Importance Sampling method.
    std::vector<std::tuple<vec, mat, cube>> proposition_gmms;
    const unsigned N_obs = fullGMM.weights.n_rows;
    const unsigned K = fullGMM.weights.n_cols;

    if (idx_gaussian == -1) // apply IMIS on all gaussians
    {
        for (size_t n = 0; n < N_obs; n++)
        {
            proposition_gmms.push_back(std::make_tuple(
                fullGMM.weights.row(n).t(),
                fullGMM.means.col(n),
                fullGMM.covs // The covariance is indenpendent from y thus it is the same for all predictions
                ));
        }
    }
    else if (idx_gaussian >= 0 && idx_gaussian < K) // apply IMIS on specific gaussian
    {

        for (size_t n = 0; n < N_obs; n++)
        {
            cube covs_cube(fullGMM.means.n_rows, fullGMM.means.n_rows, 1); // ! TODO terrible code
            covs_cube.slice(0) = fullGMM.covs.slice(idx_gaussian);
            proposition_gmms.push_back(std::make_tuple(
                vec(1, fill::value(1)),
                mat(fullGMM.means.slice(idx_gaussian).col(n)),
                covs_cube));
        }
    }
    else
    {
        throw std::invalid_argument("idx_gaussian is out of range");
    }

    return importanceSamplingParsedGMMCheck(proposition_gmms, y, y_err, N_0, B, J, covariance, verbose, seed);
}

ImportanceSamplingResult FunctionalModel::importanceSampling(const MergedGMMResult mergedGMM, const mat y, const mat y_err, const unsigned N_0, const unsigned B, const unsigned J, const vec covariance, int idx_gaussian, int verbose, unsigned seed)
{
    // retrieve the gmm parameters from de GLLiM prediction results. It corresponds to the proposition law for the Importance Sampling method.
    std::vector<std::tuple<vec, mat, cube>> proposition_gmms;
    const unsigned N_obs = mergedGMM.weights.n_rows;
    const unsigned K = mergedGMM.weights.n_cols;

    if (K == 0)
    {
        throw std::invalid_argument("mergedGMM is empty. Check k_merged parameter.");
    }

    if (idx_gaussian == -1) // apply IMIS on all gaussians
    {
        for (size_t n = 0; n < N_obs; n++)
        {
            proposition_gmms.push_back(std::make_tuple(
                mergedGMM.weights.row(n).t(),
                mergedGMM.means.col(n),
                mergedGMM.covs[n]));
        }
    }
    else if (idx_gaussian >= 0 && idx_gaussian < K) // apply IMIS on specific gaussian
    {

        for (size_t n = 0; n < N_obs; n++)
        {
            cube covs_cube(mergedGMM.means.n_rows, mergedGMM.means.n_rows, 1); // ! TODO terrible code
            covs_cube.slice(0) = mergedGMM.covs[n].slice(idx_gaussian);
            proposition_gmms.push_back(std::make_tuple(
                vec(1, fill::value(1)),
                mat(mergedGMM.means.slice(idx_gaussian).col(n)),
                covs_cube));
        }
    }
    else
    {
        throw std::invalid_argument("idx_gaussian is out of range");
    }

    return importanceSamplingParsedGMMCheck(proposition_gmms, y, y_err, N_0, B, J, covariance, verbose, seed);
}

ImportanceSamplingResult FunctionalModel::importanceSamplingParsedGMMCheck(const std::vector<std::tuple<vec, mat, cube>> &proposition_gmms, const mat y, const mat y_err, const unsigned N_0, const unsigned B, const unsigned J, const vec covariance, int verbose, unsigned seed)
{
    // TODO check sizes y, y_err,...
    // Parse the covariance argument
    const unsigned D = y.n_rows;
    vec covariance_parsed(D);
    if (covariance.n_elem == 1)
    {
        covariance_parsed.fill(covariance(0));
    }
    else if (covariance.n_elem == D)
    {
        covariance_parsed = covariance;
    }
    else
    {
        throw std::invalid_argument("covariance size must be equal to Y dimension");
    }

    // Checks on IMIS parameters
    if (J != 0 && B == 0)
    {
        throw std::invalid_argument("If IMIS steps are required (J != 0) then B must be strictly greater than 0 (B>0)");
    }
    if (B >= N_0)
    {
        throw std::invalid_argument("IMIS requires that B < N_0");
    }

    return importanceSamplingCore(proposition_gmms, y, y_err, N_0, B, J, covariance_parsed, verbose, seed);
}

ImportanceSamplingResult FunctionalModel::importanceSamplingCore(const std::vector<std::tuple<vec, mat, cube>> &proposition_gmms, const mat y, const mat y_err, const unsigned N_0, const unsigned B, const unsigned J, const vec covariance, int verbose, unsigned seed)
{
    Logger &logger = Logger::getInstance(); // set a logger with a progress bar

    unsigned N_samples = N_0 + B * J;                     // for imis
    unsigned L = std::get<1>(proposition_gmms[0]).n_rows; // get the number of rows in the first GMM mean matrix
    unsigned N_obs = y.n_cols;
    // mat results(L, N_obs);
    ImportanceSamplingResult importanceSamplingResult(L, N_obs);

    if (verbose >= 1)
    {
        logger.getProgressBar().start(N_obs);
        logger.log(INFO, "[Sampling] Start Incremental Mixture Importance Sampling (IMIS) for " + std::to_string(N_obs) + " observations.");
    }

    // You should use 'default(none)' by default: be specific about what you'resharing
    // #pragma omp parallel for default(none) schedule(static) shared(logger, importanceSamplingResult, N_samples, L, N_obs, proposition_gmms, y, y_err, covariance, N_0, B, J, verbose, seed)
    for (size_t n_obs = 0; n_obs < N_obs; n_obs++)
    {
        mat samples(L, N_samples);
        vec weights(N_samples);
        double sum_weights;
        double sum_weights_2;
        vec target_log_densities(N_samples);
        vec proposition_log_densities(N_samples);
        vec proposition_log_densities_0(B);
        gmm_full proposition_gmm;

        // ====================== Importance Sampling basic step and initialisation =========================
        proposition_gmm.set_params(std::get<1>(proposition_gmms[n_obs]), std::get<2>(proposition_gmms[n_obs]), std::get<0>(proposition_gmms[n_obs]).t());
        utils::set_seed_armadillo(seed);
        samples.cols(0, N_0 - 1) = proposition_gmm.generate(N_0);                                                                                                                                                      // sample with GLLiM-GMM
        target_log_densities.subvec(0, N_0 - 1) = targetDensity(samples.cols(0, N_0 - 1), y.col(n_obs), y_err.col(n_obs), covariance);                                                                                 // compute the target log probability density function (PDF)
        proposition_log_densities.subvec(0, N_0 - 1) = propositionDensity(samples.cols(0, N_0 - 1), std::get<0>(proposition_gmms[n_obs]), std::get<1>(proposition_gmms[n_obs]), std::get<2>(proposition_gmms[n_obs])); // compute the target log probability density function (PDF)
        weights.subvec(0, N_0 - 1) = target_log_densities.subvec(0, N_0 - 1) - proposition_log_densities.subvec(0, N_0 - 1);                                                                                           // compute weights verifying numerical stability

        // ========================================== IMIS steps ============================================

        /* The covariance matrix from the initial proposition law is used for each IMIS iteration
        The advandage is that the inverse of the covariance matrix is only computed once.
        However to improve IMIS precision the Mahalanobis distance should be calculated at each step with Covariance of each new proposition law */
        mat proposition_covariance = utils::proposition_covariance(proposition_gmm);
        proposition_covariance = trimatl(inv_sympd(proposition_covariance));

        size_t j_step, N_j, N_j1;
        std::vector<gmm_full> gmm_step(J);

        for (j_step = 0; j_step < J; j_step++)
        {
            N_j = (N_0 + j_step * B);
            N_j1 = N_j + B;

            // a) Find highest weigth
            uword i_max = weights.subvec(0, N_j - 1).index_max();
            vec x_max = samples.col(i_max);

            // b) Find the B inputs with smallest Mahalanobis distance to x_max
            vec mahalanobis_dist = utils::MahalanobisWithInvertedCov(samples.cols(0, N_j - 1), x_max, proposition_covariance);
            uvec neighboors_idx = sort_index(mahalanobis_dist);

            // d) Compute associated covariance
            /*  Raftery & Bao propose the formula
                w = (ws[id] + (1 / Nk)) / 2 (average between importance and 1/Nk)
                but, according to Fasalio et al 2016, not weighting increases stability
                w = 1*/
            cube Sigma_j(L, L, 1);
            for (unsigned id = 0; id < B; id++)
            {
                vec u_tmp = x_max - samples.col(neighboors_idx[id]);
                Sigma_j.slice(0) += u_tmp * u_tmp.t();
            }
            Sigma_j.slice(0) /= B;

            // e) Generate B new samples
            mat x_max_mat = mat(x_max);
            rowvec weight_unitary(1, fill::value(1));
            gmm_step[j_step].set_params(x_max, Sigma_j, weight_unitary); // save the Gaussian(x_max, Sigma_j) for further use ...
            utils::set_seed_armadillo(seed);
            samples.cols(N_j, N_j1 - 1) = gmm_step[j_step].generate(B);

            // h) Update proposition law
            /* Update current points [0:N_j-1]:
                for existing points, we can update the weigths without computing the whole mixture using
                prop_j1 = (N_j / N_j1) * prop_j + (B / N_j1) * phi_j1 */
            vec log_phi_j1 = gmm_step[j_step].log_p(samples.cols(0, N_j - 1)).t();
            proposition_log_densities.subvec(0, N_j - 1) = utils::weightedLogSumExp(N_j, proposition_log_densities.subvec(0, N_j - 1), B, log_phi_j1) - log(N_j1);

            /* Update new points [N_j:N_j1-1]:
                for new points, we have to compute the whole mixture (of j_step first components) using
                prop_j1 = (N_0 / N_j1) * prop_0 + (B / N_j1) * SUM(phi_j, 1:j) */
            mat log_phi_list(B, j_step + 1); // extra memory usage
            for (unsigned n = 0; n < j_step + 1; n++)
            {
                log_phi_list.col(n) = gmm_step[n].log_p(samples.cols(N_j, N_j1 - 1)).t();
            }
            vec log_sum_phi = utils::logSumExp(log_phi_list, 1);
            target_log_densities.subvec(N_j, N_j1 - 1) = targetDensity(samples.cols(N_j, N_j1 - 1), y.col(n_obs), y_err.col(n_obs), covariance);                                                             // compute the target log probability density function (PDF)
            proposition_log_densities_0 = propositionDensity(samples.cols(N_j, N_j1 - 1), std::get<0>(proposition_gmms[n_obs]), std::get<1>(proposition_gmms[n_obs]), std::get<2>(proposition_gmms[n_obs])); // compute the target log probability density function (PDF)
            proposition_log_densities.subvec(N_j, N_j1 - 1) = utils::weightedLogSumExp(N_0, proposition_log_densities_0, B, log_sum_phi) - log(N_j1);

            // i) Update all weights
            weights.subvec(0, N_j1 - 1) = target_log_densities.subvec(0, N_j1 - 1) - proposition_log_densities.subvec(0, N_j1 - 1); // Careful: here we manipulate log(weights)

        } // End of IMIS steps

        // ====================== Importance Sampling diagnostics and mean estimations ======================
        sum_weights = utils::logSumExp(weights);
        sum_weights_2 = utils::logSumExp(2 * weights);

        importanceSamplingResult.nb_effective_sample(n_obs) = uvec(find_finite(target_log_densities)).n_elem;
        importanceSamplingResult.effective_sample_size(n_obs) = exp(2 * sum_weights - sum_weights_2);
        importanceSamplingResult.qn(n_obs) = exp(weights.max() - sum_weights);

        weights -= sum_weights;
        weights = exp(weights); // back to real weights

        for (unsigned n = 0; n < N_samples; n++)
        {
            importanceSamplingResult.predictions.col(n_obs) += weights(n) * samples.col(n); // Compute samples mean
        }
        for (unsigned n = 0; n < N_samples; n++)
        {
            importanceSamplingResult.predictions_variance.col(n_obs) += weights(n) * pow(samples.col(n) - importanceSamplingResult.predictions.col(n_obs), 2); // Compute samples variance
        }
        // ==================================================================================================
        if (verbose >= 1)
            logger.getProgressBar().update(n_obs + 1);
    }

    if (verbose >= 1)
    {
        logger.getProgressBar().stop();
        logger.log(INFO, "[Sampling] IMIS completed in " + std::to_string(logger.getProgressBar().time_it_took()) + " sec");
    }

    return importanceSamplingResult;
}
