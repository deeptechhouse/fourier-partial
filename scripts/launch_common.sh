#!/bin/bash
#
# Shared launch helpers for FourierPartial.app and run.sh.
#
# This file is *sourced*, never executed. It defines functions only, sets no
# global state beyond the readonly constants below, and calls nothing at load
# time, so sourcing it twice is harmless.
#
# The two launchers need identical answers to the same three questions --
# "which port did the server actually pick?", "is it up yet?", and "how do I
# stop it?" -- so the logic lives here once rather than drifting apart in two
# copies.

# ADJUSTABLE: how long to wait for the server to answer /api/health, in seconds.
# Cold starts import numpy, scipy and FastAPI, which is a few seconds on a
# spinning disk; 30 leaves generous headroom without hanging forever.
FP_READY_TIMEOUT="${FP_READY_TIMEOUT:-30}"

# ADJUSTABLE: seconds between readiness probes.
FP_POLL_INTERVAL="${FP_POLL_INTERVAL:-0.4}"

# fp_detect_url <logfile>
#
# Echo the URL the server reported, or nothing if it has not said yet.
#
# backend/main.py prints a machine-readable "FOURIER_PARTIAL_URL=http://host:port"
# line for exactly this purpose. The bare-http:// grep is a second chance in
# case that line is ever renamed but the human-readable log line survives.
#
# Deliberately NO fallback to http://127.0.0.1:8420. The backend auto-selects a
# free port when 8420 is taken, which means 8420 being taken is precisely the
# case where guessing is wrong -- and the squatter is most likely *another
# copy of this app*, which answers /api/health happily. A speculative probe
# therefore reports success and opens a browser onto somebody else's server.
# Observed in testing; do not reintroduce the guess.
fp_detect_url() {
    local logfile="$1" url=""

    [ -f "$logfile" ] || return 0

    url=$(/usr/bin/grep -o 'FOURIER_PARTIAL_URL=http://[^[:space:]"]*' "$logfile" 2>/dev/null |
        /usr/bin/tail -n 1 | /usr/bin/cut -d= -f2-)
    if [ -z "$url" ]; then
        url=$(/usr/bin/grep -Eo 'http://(127\.0\.0\.1|localhost):[0-9]+' "$logfile" 2>/dev/null |
            /usr/bin/tail -n 1)
    fi

    [ -n "$url" ] && printf '%s\n' "$url"
    return 0
}


# fp_wait_ready <logfile> [server_pid]
#
# Poll until GET <url>/api/health succeeds. Echoes the live URL and returns 0
# on success; returns 1 on timeout and 2 if the server process died.
#
# Polling rather than sleeping a fixed time: a fixed sleep is either too short
# on a cold start (browser opens on a connection error) or wastes seconds on
# every warm start.
fp_wait_ready() {
    local logfile="$1" server_pid="${2:-}" url deadline

    deadline=$(( $(date +%s) + FP_READY_TIMEOUT ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        # A dead server will never become healthy; fail immediately so the
        # caller can show the log instead of stalling for the full timeout.
        if [ -n "$server_pid" ] && ! kill -0 "$server_pid" 2>/dev/null; then
            return 2
        fi

        # Probe only a URL the server itself reported -- never a guess.
        url=$(fp_detect_url "$logfile")
        if [ -n "$url" ] &&
            /usr/bin/curl -s -f -o /dev/null --max-time 2 "$url/api/health"; then
            printf '%s\n' "$url"
            return 0
        fi
        /bin/sleep "$FP_POLL_INTERVAL"
    done

    return 1
}


# fp_log_tail <logfile> [lines]
#
# Last few log lines, for error messages. Never fails: an empty or missing log
# is itself a useful thing to report.
fp_log_tail() {
    local logfile="$1" lines="${2:-12}"

    if [ -s "$logfile" ]; then
        /usr/bin/tail -n "$lines" "$logfile"
    else
        printf '(the server produced no output at all)\n'
    fi
}


# fp_stop_server <pid>
#
# Ask the server to shut down, then insist. SIGTERM lets FastAPI run its
# lifespan shutdown hook, which is what purges the session directory; only
# after that grace period do we escalate to SIGKILL.
fp_stop_server() {
    local pid="$1" waited=0

    [ -n "$pid" ] || return 0
    kill -0 "$pid" 2>/dev/null || return 0

    kill -TERM "$pid" 2>/dev/null || true
    while [ "$waited" -lt 50 ] && kill -0 "$pid" 2>/dev/null; do
        /bin/sleep 0.1
        waited=$((waited + 1))
    done
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
    return 0
}


# fp_reap_orphan <pidfile>
#
# Kill a server left behind by a previous run, then forget it.
#
# Why this exists: Force Quit sends SIGKILL, which no trap can catch, so the
# launcher dies while its child server keeps holding the port. Without this,
# every force-quit leaks a background python. The `ps` check makes sure the
# recorded PID has actually been recycled into *our* server and not some
# unrelated process that happens to have inherited the number.
fp_reap_orphan() {
    local pidfile="$1" pid

    [ -f "$pidfile" ] || return 0
    pid=$(/usr/bin/head -n 1 "$pidfile" 2>/dev/null | /usr/bin/tr -dc '0-9')
    rm -f "$pidfile"

    [ -n "$pid" ] || return 0
    kill -0 "$pid" 2>/dev/null || return 0
    if /bin/ps -o command= -p "$pid" 2>/dev/null | /usr/bin/grep -q 'backend.main'; then
        fp_stop_server "$pid"
    fi
    return 0
}
