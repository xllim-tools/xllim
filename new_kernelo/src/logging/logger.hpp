#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <string>
#include <mutex>
#include <memory>
#include "progressBar.hpp"

enum LogLevel
{
    INFO,
    WARNING,
    ERROR
};

/**
 * @class Logger
 * @brief A singleton logger class to handle logging with different log levels and a progress bar.
 */
class Logger
{
public:
    /**
     * @brief Gets the singleton instance of the Logger.
     * @return A reference to the singleton Logger instance.
     */
    static Logger &getInstance();

    /**
     * @brief Logs a message with a specific log level. If Progress bar is active, shows the message above the progress bar.
     * @param level The log level (INFO, WARNING, ERROR).
     * @param message The message to log.
     */
    void log(LogLevel level, const std::string &message);

    /**
     * @brief Sets and starts a new progress bar.
     * @param total The total value representing 100% completion.
     * @param width The width of the progress bar in characters.
     */
    void setProgressBar(int total, int width = 50);

    /**
     * @brief Updates the current progress bar.
     * @param value The current progress value.
     */
    void updateProgressBar(int value);

    /**
     * @brief Clears the current progress bar from the console.
     */
    void clearProgressBar();

    /**
     * @brief Stops the current progress bar from the console and print its last update.
     */
    void stopProgressBar();

    /**
     * @brief Shows the progress bar in the console.
     */
    void showProgressBar();

private:
    Logger() {}
    ~Logger() {}
    Logger(const Logger &) = delete;
    Logger &operator=(const Logger &) = delete;

    /**
     * @brief Gets the string label for a log level.
     * @param level The log level.
     * @return The string label for the log level.
     */
    std::string getLabel(LogLevel level);

    /**
     * @brief Gets the console color code for a log level.
     * @param level The log level.
     * @return The console color code for the log level.
     */
    std::string getColor(LogLevel level);

    std::mutex log_mutex;                     ///< Mutex to ensure thread-safety for logging.
    std::mutex progress_mutex;                ///< Mutex to ensure thread-safety for the progress bar.
    std::unique_ptr<ProgressBar> progressBar; ///< Pointer to the progress bar instance.
};

#endif // LOGGER_HPP
