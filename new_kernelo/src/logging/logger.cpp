#include "logger.hpp"

Logger &Logger::getInstance()
{
    static Logger instance;
    return instance;
}

void Logger::log(LogLevel level, const std::string &message)
{
    std::lock_guard<std::mutex> guard(log_mutex);
    std::string color = getColor(level);
    clearProgressBar();
    std::cout << color << getLabel(level) << ": " << message << "\033[0m" << std::endl;
    showProgressBar();
}

void Logger::setProgressBar(int total, int width)
{
    std::lock_guard<std::mutex> guard(progress_mutex);
    progressBar = std::make_unique<ProgressBar>(total, width);
    progressBar->startProgress();
}

void Logger::updateProgressBar(int value)
{
    if (progressBar && progressBar->isActive())
    {
        std::lock_guard<std::mutex> guard(progress_mutex);
        progressBar->updateProgress(value);
    }
}

void Logger::clearProgressBar()
{
    if (progressBar && progressBar->isActive())
    {
        // std::cout << "\033[2K\r"; // Clear the current line //! not working for IPython/Jupyter
        std::cout << "\r                                                                                                                  \r";
    }
}

void Logger::stopProgressBar()
{
    if (progressBar && progressBar->isActive())
    {
        progressBar->finish();
    }
}

void Logger::showProgressBar()
{
    if (progressBar && progressBar->isActive())
    {
        progressBar->display();
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
