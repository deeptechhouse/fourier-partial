#!/bin/bash
#
# fp_launch.sh — the work behind FourierPartial.app.
#
#   fp_launch.sh ensure   start the server if it is not already up; print its URL
#   fp_launch.sh stop     stop the server this project started
#
# WHY THIS IS SEPARATE FROM THE APP BUNDLE
# ----------------------------------------
# FourierPartial.app is an AppleScript applet, and it is an applet for one
# specific reason: a plain shell-script .app cannot be clicked twice.
#
# macOS will not run a bundle's executable again while that bundle is already
# running -- it sends the running instance a "reopen" event instead. A shell
# script has no way to receive that event, so the second click did nothing at
# all: no browser window, no error, no clue. (Observed: the server had been up
# for forty minutes, so every click was silently a no-op.)
#
# An applet CAN receive it, so the applet handles `on reopen` by calling
# `ensure` again. `ensure` is idempotent -- if the server is already healthy it
# just reports the URL -- so clicking the icon always ends with a browser
# window pointing at a working app, whether one was running or not.
#
# All the real logic lives here rather than in the AppleScript because shell is
# easier to read, test, and run by hand (`./run.sh` uses the same helpers).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
SESSION_DIR="$PROJECT_ROOT/sessions"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/server.log"
PID_FILE="$LOG_DIR/server.pid"

# shellcheck source=./launch_common.sh
. "$SCRIPT_DIR/launch_common.sh"

# --- Is the server we started still alive and answering? ---------------------
#
# Both halves matter. A live PID with a dead HTTP server is a hung process, and
# a healthy URL with no PID of ours belongs to somebody else -- neither counts
# as "already running" for our purposes.
fp_running_url() {
    local pid url
    [ -f "$PID_FILE" ] || return 1

    pid=$(/usr/bin/head -n 1 "$PID_FILE" 2>/dev/null | /usr/bin/tr -dc '0-9')
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1

    url="$(fp_detect_url "$LOG_FILE")"
    [ -n "$url" ] || return 1

    /usr/bin/curl -fsS -m 2 -o /dev/null "$url/api/health" 2>/dev/null || return 1
    printf '%s\n' "$url"
}

fp_ensure() {
    local url

    # Already up? Then this is a second click, and the only thing left to do is
    # hand back the URL so the caller can open a window on it.
    if url="$(fp_running_url)"; then
        printf '%s\n' "$url"
        return 0
    fi

    if [ ! -x "$PYTHON_BIN" ]; then
        printf 'MISSING_VENV\n'
        return 3
    fi

    if ! mkdir -p "$LOG_DIR" "$SESSION_DIR" 2>/dev/null; then
        printf 'NO_WRITE\n'
        return 4
    fi

    # A previous run that was Force Quit (SIGKILL cannot be trapped) may have
    # left its server holding the port.
    fp_reap_orphan "$PID_FILE"

    : >"$LOG_FILE"

    # Detached on purpose. The applet calls this through `do shell script`,
    # which waits for the command to finish -- so the server must NOT be a
    # child of this shell, or the app would hang on launch forever. nohup plus
    # a closed stdin plus redirected output means it outlives us cleanly.
    (
        cd "$PROJECT_ROOT" || exit 1
        nohup "$PYTHON_BIN" -m backend.main >>"$LOG_FILE" 2>&1 </dev/null &
        printf '%s\n' "$!" >"$PID_FILE"
    )

    local pid
    pid=$(/usr/bin/head -n 1 "$PID_FILE" 2>/dev/null | /usr/bin/tr -dc '0-9')

    url="$(fp_wait_ready "$LOG_FILE" "$pid")"
    if [ $? -ne 0 ] || [ -z "$url" ]; then
        printf 'NOT_READY\n'
        return 5
    fi

    printf '%s\n' "$url"
    return 0
}

fp_stop() {
    local pid
    [ -f "$PID_FILE" ] || return 0
    pid=$(/usr/bin/head -n 1 "$PID_FILE" 2>/dev/null | /usr/bin/tr -dc '0-9')
    fp_stop_server "$pid"
    rm -f "$PID_FILE"
    return 0
}

case "${1:-ensure}" in
    ensure) fp_ensure ;;
    stop) fp_stop ;;
    status)
        if url="$(fp_running_url)"; then
            printf 'running %s\n' "$url"
        else
            printf 'stopped\n'
            exit 1
        fi
        ;;
    *)
        printf 'usage: %s {ensure|stop|status}\n' "$(basename "$0")" >&2
        exit 2
        ;;
esac
