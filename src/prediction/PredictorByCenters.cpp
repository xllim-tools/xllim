//
// Created by reverse-proxy on 26‏/3‏/2020.
//

#include "PredictorByCenters.h"
#define DELETED true


using namespace prediction;

PredictorByCenters::PredictorByCenters(
        const std::shared_ptr<learningModel::IGLLiMLearning> &learningModel,
        unsigned k_merged,
        double threshold) {
    this->learningModel = learningModel;
    this->k_merged = k_merged;
    this->threshold = threshold;
}

void PredictorByCenters::predict(const vec &y_obs, const vec &cov_obs) {
    // compute GMM of the observation
    arma::gmm_full gmm_obs = learningModel->computeGMM(y_obs, cov_obs);
    unsigned K = gmm_obs.hefts.n_cols;
    unsigned L = gmm_obs.means.n_rows;

    // move gmms to vector<MultivariateGaussians, bool> structure
    std::vector<std::pair<MultivariateGaussian, bool>> gaussians(K);
    for(unsigned k=0; k<K; k++){
        gaussians[k].second = !DELETED;
        gaussians[k].first.weight = gmm_obs.hefts(k);
        gaussians[k].first.mean = gmm_obs.means.col(k);
        gaussians[k].first.covariance = gmm_obs.fcovs.slice(k);
    }

    // reduce the number of gaussians based on weights threshold and number of K_MERGED
    reduceGaussians(gaussians);

    while(K > k_merged){
        findPairToMerge(gaussians);
        K -= 1;
    }

    //keep only not deleted gaussians
    std::vector<MultivariateGaussian> merged_gaussians;
    for(const auto& element : gaussians){
        if(!element.second){
            merged_gaussians.push_back(element.first);
        }
    }

    // Compute the mean of the means in the mixture
    vec mean_mean_mixture(L, fill::zeros);
    for(const auto &element : merged_gaussians){
        mean_mean_mixture += element.weight * element.mean;
    }

    // Compute the mean of covariances in the mixture
    mat mean_cov_mixture(L,L,fill::zeros);
    for(const auto &element : merged_gaussians){
        mean_cov_mixture += element.covariance + element.mean * element.mean.t() * element.weight;
    }
    mean_cov_mixture -= mean_mean_mixture * mean_mean_mixture.t();

    //return std::vector<vec>();
}

MultivariateGaussian PredictorByCenters::mergeTwoGaussians(const MultivariateGaussian &g1, const MultivariateGaussian &g2) {

    MultivariateGaussian mergedG1G2;

    mergedG1G2.weight = g1.weight + g2.weight;

    double weight1_12 = g1.weight/(g1.weight + g2.weight);
    double weight2_12 = g2.weight/(g1.weight + g2.weight);

    mergedG1G2.mean = g1.mean * weight1_12 + g2.mean * weight2_12;

    mergedG1G2.covariance = g1.covariance * weight1_12 + g2.covariance * weight2_12 +
            weight1_12 * weight2_12 * (g1.mean - g2.mean) * (g1.mean - g2.mean).t();

    return mergedG1G2;
}

void PredictorByCenters::computeDissimilarityCriterion(MultivariateGaussian &g1, MultivariateGaussian &g2,
                                                       MultivariateGaussian &mergedG1G2, double &dissimilarity) {
    mergedG1G2 = mergeTwoGaussians(g1,g2);

    double det_cov_g1 = safeCovDet(g1.covariance);
    double det_cov_g2 = safeCovDet(g2.covariance);
    double det_cov_g1g2 = safeCovDet(mergedG1G2.covariance);

    dissimilarity =   0.5 * ((g1.weight + g2.weight) * log(det_cov_g1g2) -
            g1.weight * log(det_cov_g1) - g2.weight * log(det_cov_g2));

}

double PredictorByCenters::safeCovDet(mat &covariance) {
    try {
        return det(covariance);
    }catch(std::runtime_error error){
        covariance.diag() += 1e-8;
        return safeCovDet(covariance);
    }

}

void PredictorByCenters::findPairToMerge(std::vector<std::pair<MultivariateGaussian, bool>> &gaussians) {
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

void PredictorByCenters::reduceGaussians(std::vector<std::pair<MultivariateGaussian, bool>> &gaussians) {
    unsigned K = gaussians.size();

    // sort gaussians by weight in reverse direction
    std::sort(gaussians.begin(), gaussians.end(), compareByWeight);

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
        }
    }

    if(!gaussians_to_merge.empty()){
        double sum_weights = 0;
        for(const MultivariateGaussian& g : gaussians_to_merge){
            sum_weights += g.weight;
        }

        if(sum_weights != 0){
            auto it = std::find_if(gaussians.begin(), gaussians.end(),
                    [](const std::pair<MultivariateGaussian, bool>& g){
                return !g.second;
            });
            it->second = true;
            it->first.weight = sum_weights;
            //it->first.mean = mean of mixture;
            //it->first.covariance = covariance of mixture;
        }
    }
}

unsigned factorial(unsigned n){
    return (n==0 || n==1) ? 1 : factorial(n-1)*n;
}

mat PredictorByCenters::generatePermutations(unsigned N) {
    mat permutations(factorial(N), N);
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

mat PredictorByCenters::regularize(const cube &series) {
    unsigned N = series.n_slices, L = series.n_rows, K = series.n_cols;
    mat permutations = generatePermutations(K);
    mat chosenPermutations(K,N);

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




