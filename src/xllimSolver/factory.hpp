#include "gllim.hpp"

std::shared_ptr<GLLiMBase> create_gllim(unsigned K, unsigned D, unsigned L, const std::string &gamma_type, const std::string &sigma_type, unsigned n_hidden_variables = 0)
{
    if (gamma_type == "full")
    {
        if (sigma_type == "full")
        {
            return std::make_shared<GLLiM<FullCovariance, FullCovariance>>(K, D, L, gamma_type, sigma_type, n_hidden_variables);
        }
        else if (sigma_type == "diag")
        {
            return std::make_shared<GLLiM<FullCovariance, DiagCovariance>>(K, D, L, gamma_type, sigma_type, n_hidden_variables);
        }
        else if (sigma_type == "iso")
        {
            return std::make_shared<GLLiM<FullCovariance, IsoCovariance>>(K, D, L, gamma_type, sigma_type, n_hidden_variables);
        }
        else
        {
            throw std::invalid_argument("Unsupported covariance type for Sigma");
        }
    }
    else if (gamma_type == "diag")
    {
        if (sigma_type == "full")
        {
            return std::make_shared<GLLiM<DiagCovariance, FullCovariance>>(K, D, L, gamma_type, sigma_type, n_hidden_variables);
        }
        else if (sigma_type == "diag")
        {
            return std::make_shared<GLLiM<DiagCovariance, DiagCovariance>>(K, D, L, gamma_type, sigma_type, n_hidden_variables);
        }
        else if (sigma_type == "iso")
        {
            return std::make_shared<GLLiM<DiagCovariance, IsoCovariance>>(K, D, L, gamma_type, sigma_type, n_hidden_variables);
        }
        else
        {
            throw std::invalid_argument("Unsupported covariance type for Sigma");
        }
    }
    else if (gamma_type == "iso")
    {
        if (sigma_type == "full")
        {
            return std::make_shared<GLLiM<IsoCovariance, FullCovariance>>(K, D, L, gamma_type, sigma_type, n_hidden_variables);
        }
        else if (sigma_type == "diag")
        {
            return std::make_shared<GLLiM<IsoCovariance, DiagCovariance>>(K, D, L, gamma_type, sigma_type, n_hidden_variables);
        }
        else if (sigma_type == "iso")
        {
            return std::make_shared<GLLiM<IsoCovariance, IsoCovariance>>(K, D, L, gamma_type, sigma_type, n_hidden_variables);
        }
        else
        {
            throw std::invalid_argument("Unsupported covariance type for Sigma");
        }
    }
    else
    {
        throw std::invalid_argument("Unsupported covariance type for Gamma");
    }
}

std::shared_ptr<GLLiMParametersBase> create_gllim_parameters(unsigned K, unsigned D, unsigned L, const std::string &gamma_type, const std::string &sigma_type)
{
    if (gamma_type == "full")
    {
        if (sigma_type == "full")
        {
            return std::make_shared<GLLiMParametersArray<FullCovariance, FullCovariance>>(K, D, L);
        }
        else if (sigma_type == "diag")
        {
            return std::make_shared<GLLiMParametersArray<FullCovariance, DiagCovariance>>(K, D, L);
        }
        else if (sigma_type == "iso")
        {
            return std::make_shared<GLLiMParametersArray<FullCovariance, IsoCovariance>>(K, D, L);
        }
        else
        {
            throw std::invalid_argument("Unsupported covariance type for Sigma");
        }
    }
    else if (gamma_type == "diag")
    {
        if (sigma_type == "full")
        {
            return std::make_shared<GLLiMParametersArray<DiagCovariance, FullCovariance>>(K, D, L);
        }
        else if (sigma_type == "diag")
        {
            return std::make_shared<GLLiMParametersArray<DiagCovariance, DiagCovariance>>(K, D, L);
        }
        else if (sigma_type == "iso")
        {
            return std::make_shared<GLLiMParametersArray<DiagCovariance, IsoCovariance>>(K, D, L);
        }
        else
        {
            throw std::invalid_argument("Unsupported covariance type for Sigma");
        }
    }
    else if (gamma_type == "iso")
    {
        if (sigma_type == "full")
        {
            return std::make_shared<GLLiMParametersArray<IsoCovariance, FullCovariance>>(K, D, L);
        }
        else if (sigma_type == "diag")
        {
            return std::make_shared<GLLiMParametersArray<IsoCovariance, DiagCovariance>>(K, D, L);
        }
        else if (sigma_type == "iso")
        {
            return std::make_shared<GLLiMParametersArray<IsoCovariance, IsoCovariance>>(K, D, L);
        }
        else
        {
            throw std::invalid_argument("Unsupported covariance type for Sigma");
        }
    }
    else
    {
        throw std::invalid_argument("Unsupported covariance type for Gamma");
    }
}
