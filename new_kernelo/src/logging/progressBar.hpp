#ifndef PROGRESSBAR_HPP
#define PROGRESSBAR_HPP

#include <iostream>
#include <chrono>
#include <mutex>

/**
 * @class ProgressBar
 * @brief A class to display a progress bar in the console.
 */
class ProgressBar
{
public:
    /**
     * @brief Constructor for ProgressBar.
     * @param total The total value representing 100% completion.
     * @param width The width of the progress bar in characters.
     */
    ProgressBar(int total, int width = 50);

    /**
     * @brief Starts the progress bar.
     */
    void startProgress();

    /**
     * @brief Updates the progress bar to a specific value.
     * @param value The current progress value.
     */
    void updateProgress(int value);

    /**
     * @brief Finishes the progress bar and moves to the next line.
     */
    void finish();

    /**
     * @brief Checks if the progress bar is active.
     * @return True if the progress bar is active, false otherwise.
     */
    bool isActive() const;

    /**
     * @brief Gets the current progress value.
     * @return The current progress value.
     */
    int getProgress() const;

    /**
     * @brief Displays the progress bar in the console.
     */
    void display();

private:
    

    int total;                                                ///< Total value for 100% completion.
    int width;                                                ///< Width of the progress bar.
    int progress;                                             ///< Current progress value.
    bool active;                                              ///< Status of the progress bar (active or not).
    std::chrono::time_point<std::chrono::steady_clock> start; ///< Start time for the progress bar.
    std::mutex progress_mutex;                                ///< Mutex to ensure thread-safety for progress updates.
};

#endif // PROGRESSBAR_HPP
