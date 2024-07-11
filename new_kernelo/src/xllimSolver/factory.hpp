#include "gllim.hpp"

std::shared_ptr<GLLiMBase> create_gllim(unsigned L, unsigned D, unsigned K, const std::string &gamma_type, const std::string &sigma_type)
{
    if (gamma_type == "full")
    {
        if (sigma_type == "full")
        {
            return std::make_shared<GLLiM<FullCovariance, FullCovariance>>(L, D, K, gamma_type, sigma_type);
        }
        else if (sigma_type == "diag")
        {
            return std::make_shared<GLLiM<FullCovariance, DiagCovariance>>(L, D, K, gamma_type, sigma_type);
        }
        else if (sigma_type == "iso")
        {
            return std::make_shared<GLLiM<FullCovariance, IsoCovariance>>(L, D, K, gamma_type, sigma_type);
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
            return std::make_shared<GLLiM<DiagCovariance, FullCovariance>>(L, D, K, gamma_type, sigma_type);
        }
        else if (sigma_type == "diag")
        {
            return std::make_shared<GLLiM<DiagCovariance, DiagCovariance>>(L, D, K, gamma_type, sigma_type);
        }
        else if (sigma_type == "iso")
        {
            return std::make_shared<GLLiM<DiagCovariance, IsoCovariance>>(L, D, K, gamma_type, sigma_type);
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
            return std::make_shared<GLLiM<IsoCovariance, FullCovariance>>(L, D, K, gamma_type, sigma_type);
        }
        else if (sigma_type == "diag")
        {
            return std::make_shared<GLLiM<IsoCovariance, DiagCovariance>>(L, D, K, gamma_type, sigma_type);
        }
        else if (sigma_type == "iso")
        {
            return std::make_shared<GLLiM<IsoCovariance, IsoCovariance>>(L, D, K, gamma_type, sigma_type);
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

std::shared_ptr<GLLiMParametersBase> create_gllim_parameters(unsigned L, unsigned D, unsigned K, const std::string &gamma_type, const std::string &sigma_type)
{
    if (gamma_type == "full")
    {
        if (sigma_type == "full")
        {
            return std::make_shared<GLLiMParametersArma<FullCovariance, FullCovariance>>(L, D, K);
        }
        else if (sigma_type == "diag")
        {
            return std::make_shared<GLLiMParametersArma<FullCovariance, DiagCovariance>>(L, D, K);
        }
        else if (sigma_type == "iso")
        {
            return std::make_shared<GLLiMParametersArma<FullCovariance, IsoCovariance>>(L, D, K);
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
            return std::make_shared<GLLiMParametersArma<DiagCovariance, FullCovariance>>(L, D, K);
        }
        else if (sigma_type == "diag")
        {
            return std::make_shared<GLLiMParametersArma<DiagCovariance, DiagCovariance>>(L, D, K);
        }
        else if (sigma_type == "iso")
        {
            return std::make_shared<GLLiMParametersArma<DiagCovariance, IsoCovariance>>(L, D, K);
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
            return std::make_shared<GLLiMParametersArma<IsoCovariance, FullCovariance>>(L, D, K);
        }
        else if (sigma_type == "diag")
        {
            return std::make_shared<GLLiMParametersArma<IsoCovariance, DiagCovariance>>(L, D, K);
        }
        else if (sigma_type == "iso")
        {
            return std::make_shared<GLLiMParametersArma<IsoCovariance, IsoCovariance>>(L, D, K);
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
