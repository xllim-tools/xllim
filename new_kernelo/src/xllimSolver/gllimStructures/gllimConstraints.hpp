#ifndef GLLIMCONSTRAINTS_HPP
#define GLLIMCONSTRAINTS_HPP

struct GLLiMConstraints
{
    // TODO : proper definitions and documentation
    const std::string gamma_type; // The Gamma covariance matrix type ('full', 'diag', 'iso')
    const std::string sigma_type; // The Sigma covariance matrix type ('full', 'diag', 'iso')
    // ? complex formulation, latent space / hybrid gllim,

    GLLiMConstraints(const std::string &gamma_type, const std::string &sigma_type) : gamma_type(gamma_type), sigma_type(sigma_type) {}
};

#endif // GLLIMCONSTRAINTS_HPP
