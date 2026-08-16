#!/bin/bash
#
# fourier_partial -- terminal launcher.
#
# Does what FourierPartial.app does, except the server runs in the foreground
# so you can watch the log scroll by. Use this when something is broken and you
# want to see the traceback as it happens.
#
#   ./run.sh              start the server and open a browser
#   ./run.sh --no-open    start the server, do not touch the browser
#
# Press Ctrl-C to stop.
#
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
SESSION_DIR="$PROJECT_ROOT/sessions"
# The log lives OUTSIDE the session directory on purpose. The server treats
# sessions/ as its own scratch space and clears it on shutdown, so a log kept
# in there would be deleted at exactly the moment it became interesting.
LOG_DIR="$PROJECT_ROOT/logs"
# Separate log and pid from the .app's, so the two launchers cannot interfere.
# Sharing them would mean (a) starting run.sh truncates the log the running .app
# is reading, and (b) run.sh's orphan sweep would kill the .app's live server.
# Running both at once is a normal thing to do -- the backend picks a second
# port by itself -- so the two must not share mutable state.
LOG_FILE="$LOG_DIR/server-cli.log"
PID_FILE="$LOG_DIR/server-cli.pid"
COMMON_LIB="$PROJECT_ROOT/scripts/launch_common.sh"

OPEN_BROWSER=1
for arg in "$@"; do
    case "$arg" in
        --no-open) OPEN_BROWSER=0 ;;
        -h | --help)
            /usr/bin/sed -n '3,12p' "${BASH_SOURCE[0]}" | /usr/bin/sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "run.sh: unknown option '$arg' (try --help)" >&2
            exit 2
            ;;
    esac
done

# --- preflight ---------------------------------------------------------------
if [ ! -r "$COMMON_LIB" ]; then
    echo "error: missing $COMMON_LIB -- run this from inside the project checkout." >&2
    exit 1
fi
# shellcheck source=scripts/launch_common.sh
. "$COMMON_LIB"

if [ ! -x "$PYTHON_BIN" ]; then
    cat >&2 <<EOF
error: no virtualenv python at $PYTHON_BIN

Create it first:

    cd $PROJECT_ROOT
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
EOF
    exit 1
fi

mkdir -p "$LOG_DIR" "$SESSION_DIR"
: >"$LOG_FILE"

# Clear up after a previous run that was killed rather than interrupted.
fp_reap_orphan "$PID_FILE"

# --- open the browser once the server answers --------------------------------
# Runs alongside the foreground server: the server has to be running before it
# can be probed, so the probe cannot come first.
OPENER_PID=""
if [ "$OPEN_BROWSER" -eq 1 ]; then
    (
        url="$(fp_wait_ready "$LOG_FILE")"
        if [ $? -eq 0 ]; then
            echo ""
            echo "  fourier_partial is ready at $url"
            echo ""
            /usr/bin/open "$url" 2>/dev/null ||
                echo "  (could not open a browser -- visit $url yourself)"
        else
            echo ""
            echo "  fourier_partial did not answer /api/health within ${FP_READY_TIMEOUT}s."
            echo "  The server output above should say why."
            echo ""
        fi
    ) &
    OPENER_PID=$!
fi

SERVER_PID=""
cleanup() {
    trap - EXIT INT TERM
    [ -n "$OPENER_PID" ] && kill "$OPENER_PID" 2>/dev/null
    fp_stop_server "$SERVER_PID"
    rm -f "$PID_FILE"
    return 0
}
trap cleanup EXIT INT TERM

# --- run ---------------------------------------------------------------------
echo "fourier_partial: starting server (Ctrl-C to stop)"
echo "fourier_partial: log -> $LOG_FILE"
cd "$PROJECT_ROOT" || exit 1

# Process substitution rather than a `|` pipeline, for one reason: a pipeline
# gives no handle on the python process, so a SIGTERM aimed at this script
# would kill the script and leave the server running forever. This way we hold
# the real PID and the trap above can actually stop it. `tee` still puts the
# output on screen and in the log the opener is watching.
"$PYTHON_BIN" -m backend.main > >(/usr/bin/tee -a "$LOG_FILE") 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" >"$PID_FILE"

wait "$SERVER_PID"
exit $?
