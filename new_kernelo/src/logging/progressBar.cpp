#include "progressBar.hpp"

ProgressBar::ProgressBar(int total, int width)
    : total(total), width(width), progress(0), active(false), start(std::chrono::steady_clock::now()) {}

void ProgressBar::startProgress()
{
    active = true;
    progress = 0;
    start = std::chrono::steady_clock::now();
}

void ProgressBar::updateProgress(int value)
{
    std::lock_guard<std::mutex> guard(progress_mutex);
    progress = value;
    // display();
}

void ProgressBar::finish()
{
    std::lock_guard<std::mutex> guard(progress_mutex);
    active = false;
    std::cout << std::endl;
}

bool ProgressBar::isActive() const
{
    return active;
}

int ProgressBar::getProgress() const
{
    return progress;
}

void ProgressBar::display()
{
    if (active)
    {
        float percentage = static_cast<float>(progress) / total;
        int pos = width * percentage;

        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = now - start;

        std::cout << "\033[35m"; // set color to magenta
        // std::cout << "\r[";
        std::cout << "\r[";
        for (int i = 0; i < width; ++i)
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
