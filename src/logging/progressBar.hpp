#ifndef PROGRESS_BAR_HPP
#define PROGRESS_BAR_HPP

#include "timer.hpp"

#include <iostream>
#include <thread>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#else
/// Macros used to disguise the fact that we do not have multithreading enabled.
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

///@brief Manages a console-based progress bar to keep the user entertained.
/// Defining the global `NOPROGRESS` will
/// disable all progress operations, potentially speeding up a program. The look
/// of the progress bar is shown in ProgressBar.hpp.
class ProgressBar
{
public:
    ProgressBar() : active(false) {}

    /// @brief Start/reset the progress bar.
    /// @param total_work  The amount of work to be completed, usually specified in cells.
    void start(uint32_t total_work)
    {
        active = true;
        timer = Timer();
        timer.start();
        this->total_work = total_work;
        next_update = 0;
        call_diff = total_work / 200;
        old_percent = 0;
        work_done = 0;
        clearConsoleLine();
    }

    ///@brief Update the visible progress bar, but only if enough work has been done.
    ///
    /// Define the global `NOPROGRESS` flag to prevent this from having an
    /// effect. Doing so may speed up the program's execution.
    void update(uint32_t work_done0)
    {
// Provide simple way of optimizing out progress updates
#ifdef NOPROGRESS
        return;
#endif

        // Quick return if this isn't the main thread
        if (omp_get_thread_num() != 0)
            return;

        // Update the amount of work done
        // work_done0 sould be in [1; total_work].
        work_done = work_done0;

        // Quick return if insufficient progress has occurred
        if (work_done < next_update)
            return;

        // Update the next time at which we'll do the expensive update stuff
        next_update += call_diff;

        // Use a uint16_t because using a uint8_t will cause the result to print as a
        // character instead of a number
        uint16_t percent = (uint8_t)(work_done * omp_get_num_threads() * 100 / total_work);

        // Handle overflows
        if (percent > 100)
            percent = 100;

        // In the case that there has been no update (which should never be the case,
        // actually), skip the expensive screen print
        if (percent == old_percent)
            return;

        // Update old_percent accordingly
        old_percent = percent;

        show();
    }

    // Print an update string which looks like this:
    //   [================================================  ] (96% - 1.0s - 4 threads)
    void show()
    {
        std::cout << "\r\033[2K["
                  << "\033[35m"
                  << std::string(old_percent / 2, '=') << std::string(50 - old_percent / 2, ' ')
                  << "] ("
                  << old_percent << "% - "
                  << std::fixed << std::setprecision(1) << timer.lap() / old_percent * (100 - old_percent)
                  << "s - "
                  << omp_get_num_threads() << " threads)" << std::flush;
    }

    bool isActive()
    {
        return active;
    }

    /// Increment by one the work done and update the progress bar
    ProgressBar &operator++()
    {
        // Quick return if this isn't the main thread
        if (omp_get_thread_num() != 0)
            return *this;

        work_done++;
        update(work_done);
        return *this;
    }

    /// Stop the progress bar. Throws an exception if it wasn't started.
    ///@return The number of seconds the progress bar was running.
    void stop()
    {
        clearConsoleLine();
        active = false;
        timer.stop();
        // return timer.accumulated();
    }

    ///@return Return the time the progress bar ran for.
    double time_it_took()
    {
        return timer.accumulated();
    }

    uint32_t cellsProcessed() const
    {
        return work_done;
    }

private:
    bool active;
    uint32_t total_work;  // Total work to be accomplished
    uint32_t next_update; // Next point to update the visible progress bar
    uint32_t call_diff;   // Interval between updates in work units
    uint32_t work_done;
    uint16_t old_percent; // Old percentage value (aka: should we update the progress bar) TODO: Maybe that we do not need this
    Timer timer;          // Used for generating ETA

    /// Clear current line on console so a new progress bar can be written
    void clearConsoleLine() const
    {
        std::cout << "\r\033[2K" << std::flush;
    }
};

#endif // PROGRESS_BAR_HPP
