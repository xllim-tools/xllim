#include "logger.hpp"

Logger &Logger::getInstance()
{
    static Logger instance;
    return instance;
}

Logger::Logger() : progress_bar_active_(false) {}

void Logger::log(LogLevel level, const std::string &message)
{
    std::lock_guard<std::mutex> guard(log_mutex_);
    clearProgressBar();
    std::cout << getLevelColor(level) << "[xllim] " << getLevelLabel(level) << ": " << message << getColor("reset") << std::endl;
    showProgressBar();
}

void Logger::log(LogLevel level, unsigned verbose_level, unsigned verbose, const std::string &message)
{
    if (verbose >= verbose_level)
    {
        std::lock_guard<std::mutex> guard(log_mutex_);
        clearProgressBar();
        std::cout << getLevelColor(level) << "[xllim] " << getLevelLabel(level) << ": " << message << getColor("reset") << std::endl;
        showProgressBar();
    }
}

void Logger::startProgressBar(int total, int width)
{
    progress_bar_active_ = true;
    progress_bar_total_ = total;
    progress_bar_width_ = width;
    progress_bar_progress_ = 0;
    progress_bar_start_ = std::chrono::steady_clock::now();
}

void Logger::updateProgressBar(int value)
{
    if (progress_bar_active_)
    {
        std::lock_guard<std::mutex> guard(log_mutex_);
        progress_bar_progress_ = value;
    }
}

void Logger::clearProgressBar()
{
    if (progress_bar_active_)
    {
        // std::cout << "\033[2K\r"; // Clear the current line //! not working for IPython/Jupyter
        std::cout << "\r                                                                                                                  \r";
    }
}

void Logger::stopProgressBar()
{
    if (progress_bar_active_)
    {
        std::lock_guard<std::mutex> guard(log_mutex_);
        progress_bar_active_ = false;
        std::cout << std::endl;
    }
}

void Logger::showProgressBar()
{
    if (progress_bar_active_)
    {
        float percentage = static_cast<float>(progress_bar_progress_) / progress_bar_total_;
        int pos = progress_bar_width_ * percentage;

        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = now - progress_bar_start_;

        std::string progress_bar_string;
        progress_bar_string += getColor("magenta"); // set color to magenta
        progress_bar_string += "\r[";               // carriage return. Cursur is placed at the start of the line
        for (int i = 0; i < progress_bar_width_; ++i)
        {
            if (i < pos)
                progress_bar_string += "=";
            else if (i == pos)
                progress_bar_string += ">";
            else
                progress_bar_string += " ";
        }
        progress_bar_string += "] " + std::to_string(int(percentage * 100.0)) + "% (" + std::to_string(elapsed_seconds.count()) + " sec)";
        progress_bar_string += getColor("reset"); // reset color
        std::cout << progress_bar_string;
        std::cout.flush();
    }
}

std::string Logger::getLevelLabel(LogLevel level)
{
    switch (level)
    {
    case INFO:
        return "INFO    ";
    case WARNING:
        return "WARNING ";
    case ERROR:
        return "ERROR   ";
    default:
        return "UNKNOWN ";
    }
}

std::string Logger::getLevelColor(LogLevel level)
{
    switch (level)
    {
    case INFO:
        return getColor("default");
    case WARNING:
        return getColor("yellow");
    case ERROR:
        return getColor("red");
    default:
        return getColor("reset");
    }
}

std::string Logger::getColor(std::string color)
{
    if (color == "default")
        return "\033[39m"; // Default
    else if (color == "yellow")
        return "\033[33m"; // Yellow
    else if (color == "red")
        return "\033[31m"; // Red
    else if (color == "magenta")
        return "\033[35m"; // Magenta
    else if (color == "reset")
        return "\033[0m"; // Reset
    else
        return "\033[0m"; // Reset
}
