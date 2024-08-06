#include "logger.hpp"

Logger &Logger::getInstance()
{
    static Logger instance;
    return instance;
}

Logger::Logger() : progress_bar_active_(false) {}

void Logger::log(LogLevel level, const std::string &message)
{
    std::lock_guard<std::mutex> guard(log_mutex);
    clearProgressBar();
    std::cout << getColor(level) << getLabel(level) << ": " << message << "\033[0m" << std::endl;
    showProgressBar();
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
        std::lock_guard<std::mutex> guard(log_mutex);
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
        std::lock_guard<std::mutex> guard(log_mutex);
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

        std::cout << "\033[35m"; // set color to magenta
        std::cout << "\r[";      // carriage return. Cursur is placed at the start of the line
        for (int i = 0; i < progress_bar_width_; ++i)
        {
            if (i < pos)
                std::cout << "=";
            else if (i == pos)
                std::cout << ">";
            else
                std::cout << " ";
        }
        std::cout << "] " << int(percentage * 100.0) << "% (" << elapsed_seconds.count() << "s)";
        std::cout << "\033[0m"; // reset color
        std::cout.flush();
    }
}

std::string Logger::getLabel(LogLevel level)
{
    switch (level)
    {
    case INFO:
        return "INFO";
    case WARNING:
        return "WARNING";
    case ERROR:
        return "ERROR";
    default:
        return "UNKNOWN";
    }
}

std::string Logger::getColor(LogLevel level)
{
    switch (level)
    {
    case INFO:
        return "\033[39m"; // Default
    case WARNING:
        return "\033[33m"; // Yellow
    case ERROR:
        return "\033[31m"; // Red
    default:
        return "\033[0m"; // Reset
    }
}
