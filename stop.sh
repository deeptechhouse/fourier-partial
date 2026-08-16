#!/bin/bash
#
# stop.sh — stop the fourier_partial server.
#
# Normally you do not need this: quitting FourierPartial.app stops the server,
# and ./run.sh stops it on Ctrl-C. It is here for the case where the app was
# Force Quit (SIGKILL cannot be trapped, so nothing gets to clean up) and its
# server is still holding the port.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
./scripts/fp_launch.sh stop
echo "fourier_partial stopped."
