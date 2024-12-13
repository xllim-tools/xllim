#ifndef LOGGER_HPP
#define LOGGER_HPP

#include "progressBar.hpp"

#include <iostream>
#include <string>
#include <mutex>

enum LogLevel
{
    INFO,
    WARNING,
    ERROR
};

/// @class Logger
/// @brief A singleton logger class to handle logging with different log levels and a progress bar.
class Logger
{
public:
    // Delete copy constructor and assignment operator
    Logger(const Logger &) = delete;
    Logger &operator=(const Logger &) = delete;

    /// @brief Gets the singleton instance of the Logger.
    /// @return A reference to the singleton Logger instance.
    static Logger &getInstance()
    {
        static Logger instance;
        return instance;
    };

    /// @brief Logs a message to std::cout with a specific log level.
    /// @param level The log level (INFO, WARNING, ERROR).
    /// @param message The message to log.
    void log(LogLevel level, const std::string &message)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // std::cout << "\r\033[2K" << std::flush; // It is not well handle by Django (PlanetGLLiM) but it my be useful for logging with an active ProgressBar ...
        std::cout << "\r" << getLevelColor(level) << "[xllim] " << getLevelLabel(level) << ": " << message << getColor("reset") << std::endl;
    };

    /// @brief Gets the `ProgressBar` object associated with the singleton Logger.
    /// @return The reference to `ProgressBar` instance.
    ProgressBar &getProgressBar()
    {
        return bar_;
    };

private:
    Logger() : bar_() {};
    ~Logger() {};

    /// @brief Gets the string label for a log level.
    /// @param level The log level.
    /// @return The string label for the log level.
    std::string getLevelLabel(LogLevel level)
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
    };

    /// @brief Gets the console color code for a log level.
    /// @param level The log level.
    /// @return The console color code for the log level.
    std::string getLevelColor(LogLevel level)
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
    };

    /// @brief Gets the ANSI escape code of a color
    /// @param color A string of the color.
    /// @return The ANSI escape code color
    std::string getColor(std::string color)
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
    };

    ProgressBar bar_;  // `ProgressBar` unique object
    std::mutex mutex_; // Mutex to ensure thread-safety for logging.
};

#endif // LOGGER_HPP
