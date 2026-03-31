#include "deployment_common.hpp"

std::atomic_bool g_stop_requested = false;

void handle_sigint(int) {
    if (!g_stop_requested.exchange(true)) {
        const char msg[] = "\nSIGINT received, waiting to finish current test...\n";
        write(2, msg, sizeof msg - 1);
    }
}
