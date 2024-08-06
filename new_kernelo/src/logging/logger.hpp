#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <string>
#include <mutex>
#include <memory>

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
    void startProgressBar(int total, int width = 50);

    /**
     * @brief Updates the current progress bar.
     * @param value The current progress value.
     */
    void updateProgressBar(int value);

    /**
     * @brief Stops the current progress bar from the console and print its last update.
     */
    void stopProgressBar();

    /**
     * @brief Shows the progress bar in the console.
     */
    void showProgressBar();

private:
    Logger();
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

    /**
     * @brief Clears the current progress bar from the console.
     */
    void clearProgressBar();

    std::mutex log_mutex;                                                   ///< Mutex to ensure thread-safety for logging.
    int progress_bar_total_;                                                ///< Total value for 100% completion.
    int progress_bar_width_;                                                ///< Width of the progress bar.
    int progress_bar_progress_;                                             ///< Current progress value.
    bool progress_bar_active_;                                              ///< Status of the progress bar (active or not).
    std::chrono::time_point<std::chrono::steady_clock> progress_bar_start_; ///< Start time for the progress bar.
};

#endif // LOGGER_HPP
