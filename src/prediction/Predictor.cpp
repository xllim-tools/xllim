/**
 * @file Predictor.cpp
 * @brief Predictor class implementation
 * @author Sami DJOUADI
 * @version 1.2
 * @date 26/03/2019
 */

#include "Predictor.h"
#include "../logging/Logger.h"

#define DELETED true


using namespace prediction;

Predictor::Predictor(
        const std::shared_ptr<learningModel::IGLLiMLearning> &learningModel,
        unsigned k_merged,
        unsigned k_pred_mean,
        double threshold) {
    this->learningModel = learningModel;
    this->k_merged = k_merged;
    this->k_pred_mean = k_pred_mean;
    this->threshold = threshold;
}

bool compareByWeight(const std::pair<MultivariateGaussian, bool> &g1, const std::pair<MultivariateGaussian, bool> &g2){
    return g1.first.weight > g2.first.weight;
}


PredictionResult Predictor::predict(const vec &y_obs, const vec &cov_obs) {
    PredictionResult result;
    // compute GMM of the observation
    arma::gmm_full gmm_obs = learningModel->computeGMM(y_obs, cov_obs);

    unsigned K = gmm_obs.means.n_cols;
    unsigned L = gmm_obs.means.n_rows;
    std::vector<std::pair<vec,mat>> predicitons;

    // move gmms to vector<MultivariateGaussians, bool> structure
    std::vector<std::pair<MultivariateGaussian, bool>> gaussians(K);
    for(unsigned k=0; k<K; k++){
        gaussians[k].second = !DELETED;
        gaussians[k].first.weight = gmm_obs.hefts(k);
        gaussians[k].first.mean = gmm_obs.means.col(k);
        gaussians[k].first.covariance = gmm_obs.fcovs.slice(k);
    }

    // Get the k_gaussian_mean best gaussians for prediction by mean
    std::sort(gaussians.begin(), gaussians.end(), compareByWeight);
    std::vector<MultivariateGaussian> gaussians_for_predi_mean;

//    for(unsigned k=0; k<k_pred_mean; k++){
//        gaussians_for_predi_mean.push_back(gaussians[k].first);
//    }

    // reduce the number of gaussians based on weights threshold and number of K_MERGED
    //reduceGaussians(gaussians, K);

    while(K >= k_merged){
        if (K == k_pred_mean){
            for(const auto& element : gaussians){
                if(!element.second){
                    gaussians_for_predi_mean.push_back(element.first);
                }
            }
        }
        if(K > k_merged){
            findPairToMerge(gaussians);
        }
        K -= 1;
    }

    result.centerPredResult.weights = vec(k_merged);
    result.centerPredResult.means = mat(L,k_merged);
    result.centerPredResult.covs = cube(L,L,k_merged);
    result.meanPredResult.gmm_weights = vec(k_pred_mean);
    result.meanPredResult.gmm_means = mat(L,k_pred_mean);
    result.meanPredResult.gmm_covs = cube(L,L,k_pred_mean);

    //keep only not deleted gaussians
    unsigned i=0;
    for(const auto& element : gaussians){
        if(!element.second){
            result.centerPredResult.weights(i) = element.first.weight;
            i++;
            predicitons.emplace_back(element.first.mean,element.first.covariance);
        }
    }
    // Compute the mean of the means in the mixture
    result.meanPredResult.mean = vec(L, fill::zeros);
    for(const auto &element : gaussians_for_predi_mean){
        result.meanPredResult.mean += element.weight * element.mean;
    }

    // Compute the mean of covariances in the mixture
    mat mean_cov_mixture(L,L,fill::zeros);
    for(const auto &element : gaussians_for_predi_mean){
        mean_cov_mixture += (element.covariance + element.mean * element.mean.t()) * element.weight;
    }
    mean_cov_mixture -= result.meanPredResult.mean * result.meanPredResult.mean.t();
    result.meanPredResult.variance = mean_cov_mixture.diag();


    for(unsigned k=0; k<k_pred_mean; k++){
        result.meanPredResult.gmm_weights(k) = gaussians_for_predi_mean[k].weight;
        result.meanPredResult.gmm_means.col(k) = gaussians_for_predi_mean[k].mean;
        result.meanPredResult.gmm_covs.slice(k) = gaussians_for_predi_mean[k].covariance;
    }

    for(unsigned k=0; k<k_merged; k++){
        result.centerPredResult.means.col(k) = predicitons[k].first;
        result.centerPredResult.covs.slice(k) = predicitons[k].second;
    }

    return result;
}

MultivariateGaussian Predictor::mergeTwoGaussians(const MultivariateGaussian &g1, const MultivariateGaussian &g2) {

    MultivariateGaussian mergedG1G2;

    mergedG1G2.weight = g1.weight + g2.weight;

    double weight1_12 = g1.weight/(g1.weight + g2.weight);
    double weight2_12 = g2.weight/(g1.weight + g2.weight);

    mergedG1G2.mean = g1.mean * weight1_12 + g2.mean * weight2_12;

    mergedG1G2.covariance = g1.covariance * weight1_12 + g2.covariance * weight2_12 +
            weight1_12 * weight2_12 * (g1.mean - g2.mean) * (g1.mean - g2.mean).t();

    return mergedG1G2;
}

void Predictor::computeDissimilarityCriterion(MultivariateGaussian &g1, MultivariateGaussian &g2,
                                              MultivariateGaussian &mergedG1G2, double &dissimilarity) {
    mergedG1G2 = mergeTwoGaussians(g1,g2);

    double det_cov_g1 = safeCovDet(g1.covariance);
    double det_cov_g2 = safeCovDet(g2.covariance);
    double det_cov_g1g2 = safeCovDet(mergedG1G2.covariance);

    dissimilarity =   0.5 * ((g1.weight + g2.weight) * log(det_cov_g1g2) -
            g1.weight * log(det_cov_g1) - g2.weight * log(det_cov_g2));

}

double Predictor::safeCovDet(mat &covariance) {
    try {
        return det(covariance);
    }catch(const std::runtime_error &error){
        covariance.diag() += 1e-8;
        return safeCovDet(covariance);
    }

}

void Predictor::findPairToMerge(std::vector<std::pair<MultivariateGaussian, bool>> &gaussians) {
    unsigned K = gaussians.size();
    MultivariateGaussian proposition_merge, best_merge;
    unsigned best_k1, best_k2;
    double proposition_d, best_d = datum::inf;

    for(unsigned k1=0; k1<K; k1++){
        if(gaussians[k1].second != DELETED){
            for(unsigned k2=k1+1; k2<K; k2++){
                if(gaussians[k2].second != DELETED){
                    computeDissimilarityCriterion(
                            gaussians[k1].first,
                            gaussians[k2].first,
                            proposition_merge,
                            proposition_d);
                    if(proposition_d < best_d){
                        best_k1 = k1;
                        best_k2 = k2;
                        best_d = proposition_d;
                        best_merge = proposition_merge;
                    }
                }
            }
        }
    }
    gaussians[best_k1].first = best_merge;
    gaussians[best_k2].second = DELETED;

}


vec computeMixtureMean(const vec &weights, const mat &means) {
    vec barycenter(means.n_rows, fill::zeros);
    for(unsigned k=0; k<weights.n_rows; k++){
        barycenter += means.col(k) * weights(k);
    }
    return barycenter;
}

mat computeMixtureCov(const vec &weights, const mat &means, const cube &covs) {
    unsigned L = means.n_rows;
    vec mean_mean_mixture = computeMixtureMean(weights, means);
    mat mean_cov_mixture(L,L,fill::zeros);

    for(unsigned k=0; k<weights.n_rows; k++){
        mean_cov_mixture += covs.slice(k) + means.col(k)  * means.col(k) .t() * weights(k);
    }
    mean_cov_mixture -= mean_mean_mixture * mean_mean_mixture.t();

    return mean_cov_mixture;
}

void Predictor::reduceGaussians(std::vector<std::pair<MultivariateGaussian, bool>> &gaussians, unsigned &K) {
    unsigned L = gaussians[0].first.mean.n_rows;
    unsigned reduced_K = k_merged;

    // sort gaussians by weight in reverse direction
    //std::sort(gaussians.begin(), gaussians.end(), compareByWeight);

    std::vector<MultivariateGaussian> gaussians_to_merge;

    for(unsigned k=0; k<k_merged; k++){
        gaussians[k].second = !DELETED;
    }

    for(unsigned k=k_merged; k<K; k++){
        if(gaussians[k].first.weight < threshold){
            gaussians_to_merge.push_back(gaussians[k].first);
            gaussians[k].second = DELETED;
        }else{
            gaussians[k].second = !DELETED;
            reduced_K++;
        }
    }

    if(!gaussians_to_merge.empty()){
        double sum_weights = 0;
        vec weights(gaussians_to_merge.size());
        mat means(L,gaussians_to_merge.size());
        cube covs(L,L,gaussians_to_merge.size());
        for(unsigned i=0; i<gaussians_to_merge.size(); i++){
            sum_weights += gaussians_to_merge[i].weight;
            weights(i) = gaussians_to_merge[i].weight;
            means.col(i) =  gaussians_to_merge[i].mean;
            covs.slice(i) = gaussians_to_merge[i].covariance;
        }

        if(sum_weights != 0){
            auto it = std::find_if(gaussians.begin(), gaussians.end(),
                    [](const std::pair<MultivariateGaussian, bool>& g){
                return g.second;
            });
            it->second = false;
            it->first.weight = sum_weights;
            it->first.mean = computeMixtureMean(weights, means);
            it->first.covariance = computeMixtureCov(weights, means, covs);
        }
        reduced_K++;
    }
    K = reduced_K;
}

unsigned factorial(unsigned n){
    return (n==0 || n==1) ? 1 : factorial(n-1)*n;
}

Mat<unsigned> Predictor::generatePermutations(unsigned N) {
    Mat<unsigned> permutations(factorial(N), N);
    unsigned currentPerm = 0;

    // Generate an 1D array of N elements from 0 to N-1
    unsigned elements[N];
    for(unsigned n=0; n<N; n++)
        elements[n] = n;

    do{
        // save the current permutation
        for(unsigned n=0; n<N; n++)
            permutations(currentPerm, n) = elements[n];

        currentPerm++;
    }while (std::next_permutation(elements, elements+ N));

    return permutations;
}

Mat<unsigned> Predictor::regularize(const cube &series) {
    unsigned N = series.n_slices, L = series.n_rows, K = series.n_cols;
    Mat<unsigned> permutations = generatePermutations(K);
    Mat<unsigned> chosenPermutations(K,N);

    for(unsigned k=0; k<K; k++)
        chosenPermutations(k,0) = k;

    mat currentChoice = series.slice(0);
    mat proposition(L,K);
    vec diff(permutations.n_rows);

    uword bestPermutationIndex;

    for(unsigned n=0; n<N-1; n++){
        for(unsigned p=0; p<permutations.n_rows; p++){
           for(unsigned k=0; k<K; k++){
               proposition.col(k) = series.slice(n+1).col(permutations(p,k));
           }

           diff(p) = sum(sqrt(sum(pow(currentChoice - proposition, 2), 0)));
        }
        bestPermutationIndex = diff.index_min();
        chosenPermutations.col(n+1) = permutations.row(bestPermutationIndex).t();

        for(unsigned k=0; k<K; k++){
            currentChoice.col(k) = series.slice(n+1).col(permutations(bestPermutationIndex,k));
        }
    }

    return chosenPermutations;
}

vec Predictor::computeMixtureMean(const vec &weights, const mat &means) {
    vec barycenter(means.n_rows, fill::zeros);
    for(unsigned k=0; k<weights.n_rows; k++){
        barycenter += means.col(k) * weights(k);
    }
    return barycenter;
}

mat Predictor::computeMixtureCov(const vec &weights, const mat &means, const cube &covs) {
    unsigned L = means.n_rows;
    vec mean_mean_mixture = computeMixtureMean(weights, means);
    mat mean_cov_mixture(L,L,fill::zeros);

    for(unsigned k=0; k<weights.n_rows; k++){
        mean_cov_mixture += covs.slice(k) + means.col(k)  * means.col(k) .t() * weights(k);
    }
    mean_cov_mixture -= mean_mean_mixture * mean_mean_mixture.t();

    return mean_cov_mixture;
}




