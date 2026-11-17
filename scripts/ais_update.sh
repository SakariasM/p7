#!/bin/bash
# The daily AIS update on the hosting machine: add the days the DMA has published since the
# data on disk was made. The DMA puts day D at http://aisdata.ais.dk/aisdk-D.zip around
# 00:20 UTC on D+3, so the newest day is always three days old. p7-ais-update.timer runs this
# every night; a day the DMA is late with is picked up by a later run.
#   scripts/ais_update.sh                          # every missing day up to three days ago
#   scripts/ais_update.sh 2026-10-01 2026-11-08    # the missing days of this period (a gap)
#
# Each day is built in a staging folder, data/staging/build (a project folder of its own):
#   ais process            DMA zip -> processed/ais/date=D/ais.parquet         harness: ais
#   ais clean-processed    -> clean/{positions,vessels,quality}/date=D/        map
#   clean_positions.py     -> processed/ais_clean/date=D/ais.parquet           harness: vessel_positions
#   build_vessel_days.py   -> derived/vessel_days/date=D/vessel_days.parquet   harness: vessel_days
# and moved into data/ only when all four are there, processed/ais last: the harness takes its
# days from processed/ais, and with a day there that lacks one of the other two it falls back
# to the raw positions or drops the vessel_days view (ais_tool.py). A day already in data/ is
# never replaced; a build that failed is started again by the next run.
#
# After the last day: vessel_static, a restart of p7-harness (it reads the days when it
# starts), `ais derive` (trips and stops, always from all of clean/) and a restart of p7-map.
# If one of these fails, the next run does them again, also when it adds no day.
#
# Environment:
#   NO_PUBLISH=1   build into staging and stop there: nothing in data/ changes
#   NO_DERIVE=1    leave trips and stops as they are (a later run without it rebuilds them)
#   NO_RESTART=1   don't restart the services
#   AIS_THREADS, AIS_MEMORY_LIMIT   DuckDB limits per step (default 8 and 16GB)
set -u -o pipefail
umask 002
export PYTHONUNBUFFERED=1
export P7_PROJECT_DIR="${P7_PROJECT_DIR:-/srv/p7}"
export AIS_THREADS="${AIS_THREADS:-8}" AIS_MEMORY_LIMIT="${AIS_MEMORY_LIMIT:-16GB}"
REPO="$(dirname "$(dirname "$(readlink -f "$0")")")"
AIS="$REPO/.venv/bin/ais"
PY="$P7_PROJECT_DIR/envs/harness/bin/python"
DATA="$P7_PROJECT_DIR/data"
STAGE="$DATA/staging/build"
SDATA="$STAGE/data"
PENDING="$DATA/staging/.pending"   # days were added and the steps after them haven't all run
DMA_URL="http://aisdata.ais.dk"
MIN_FREE_GB=20

# Ctrl-C or `systemctl stop` ends the whole run, not only the step that was running.
trap 'echo "$(date +%T) interrupted"; exit 130' INT TERM

if [ $# -ne 0 ] && [ $# -ne 2 ]; then
    sed -n '2,7p' "$0"
    exit 2
fi
mkdir -p "$STAGE/tmp" "$P7_PROJECT_DIR/logs/ais_update"
exec > >(tee -a "$P7_PROJECT_DIR/logs/ais_update/$(date +%F).log") 2>&1
exec 9> "$DATA/staging/.lock"
if ! flock -n 9; then
    echo "$(date +%T) another ais_update is running; nothing done"
    exit 0
fi
echo "=== $(date '+%F %T') ais_update $*"
free_gb=$(df --output=avail -BG "$DATA" | tail -1 | tr -dc 0-9)
if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
    echo "only ${free_gb}G free under $DATA (a day takes about 0.5G, a build more): nothing done"
    exit 1
fi

if [ $# -eq 2 ]; then
    first=$(date -u -d "$1" +%F) && last=$(date -u -d "$2" +%F) || exit 2
else
    first=$(ls "$DATA/processed/ais" | sed -n 's/^date=//p' | sort | head -1)
    last=$(date -u -d '3 days ago' +%F)
fi
[ -n "$first" ] || { echo "no days under $DATA/processed/ais to start from"; exit 1; }

in_ais() { AIS_DATA_DIR="$SDATA" AIS_PROCESSED_DIR="$SDATA/processed/ais" "$AIS" "$@"; }
in_harness() { (cd "$REPO/harness" && P7_PROJECT_DIR="$1" "$PY" -u "${@:2}" \
    --threads "$AIS_THREADS" --memory "$AIS_MEMORY_LIMIT" --spill "$STAGE/tmp"); }

staged() {  # day: all of its files are in staging
    local t
    for t in positions vessels quality; do
        [ -e "$SDATA/clean/$t/date=$1/part.parquet" ] || return 1
    done
    [ -e "$SDATA/processed/ais/date=$1/ais.parquet" ] &&
        [ -e "$SDATA/processed/ais_clean/date=$1/ais.parquet" ] &&
        [ -e "$SDATA/derived/vessel_days/date=$1/vessel_days.parquet" ]
}

build() {  # day: the four steps, each of which skips what it has already made
    in_ais process "$1" &&
        in_ais clean-processed "$1" "$1" --no-build &&
        in_harness "$STAGE" clean_positions.py &&
        in_harness "$STAGE" build_vessel_days.py &&
        staged "$1"
}

move() {  # staged path, path in data/: what is already in data/ stays
    if [ -e "$2" ]; then
        rm -rf "$1"
    else
        mkdir -p "$(dirname "$2")" && mv -T "$1" "$2"
    fi
}

publish() {  # day
    local t
    for t in positions vessels quality; do
        move "$SDATA/clean/$t/date=$1" "$DATA/clean/$t/date=$1" || return 1
    done
    # The counts file is written after the day's file, so it can be missing after a crash.
    if [ -e "$SDATA/processed/ais_clean/_quality/$1.json" ]; then
        move "$SDATA/processed/ais_clean/_quality/$1.json" \
            "$DATA/processed/ais_clean/_quality/$1.json" || return 1
    fi
    move "$SDATA/processed/ais_clean/date=$1" "$DATA/processed/ais_clean/date=$1" &&
        move "$SDATA/derived/vessel_days/date=$1" "$DATA/derived/vessel_days/date=$1" &&
        move "$SDATA/processed/ais/date=$1" "$DATA/processed/ais/date=$1"
}

restart() {  # unit, health URL: restart it if it runs, then wait until it answers
    [ -z "${NO_RESTART:-}" ] || return 1
    if systemctl is-active -q p7-base; then
        echo "p7-base is running (a test of the base model), so $1 is not restarted"
        return 1
    fi
    sudo -n systemctl try-restart "$1" || return 1
    systemctl is-active -q "$1" || { echo "$1 is not running, so it was not started"; return 0; }
    for _ in $(seq 60); do
        sleep 5
        curl -s -m 5 "$2" | grep -q '"ok"' && { echo "$(date +%T) restarted $1"; return 0; }
    done
    echo "$1 doesn't answer on $2 five minutes after its restart"
    return 1
}

added=0 failed=0 waiting=0
day=$first
while [[ ! "$day" > "$last" ]]; do
    d=$day
    day=$(date -u -d "$d + 1 day" +%F)
    [ -e "$DATA/processed/ais/date=$d/ais.parquet" ] && continue
    if ! staged "$d"; then
        if [ ! -e "$SDATA/processed/ais/date=$d/ais.parquet" ]; then
            code=$(curl -s -o /dev/null -m 60 -I -w '%{http_code}' "$DMA_URL/aisdk-$d.zip")
            if [ "$code" = 404 ]; then
                echo "$d: not published by the DMA (yet)"
                waiting=$((waiting + 1))
                continue
            elif [ "$code" != 200 ]; then
                echo "$d: FAILED, the DMA server answered $code"
                failed=$((failed + 1))
                continue
            fi
        fi
        echo "--- $(date +%T) $d"
        if ! build "$d"; then
            echo "$d: FAILED"
            failed=$((failed + 1))
            continue
        fi
    fi
    if [ -n "${NO_PUBLISH:-}" ]; then
        echo "$d: staged"
    elif publish "$d"; then
        echo "$d: added"
        added=$((added + 1))
        touch "$PENDING"
    else
        echo "$d: FAILED to move into $DATA"
        failed=$((failed + 1))
    fi
done
echo "=== $(date +%T) $added day(s) added, $failed failed, $waiting not published yet"
if [ -e "$SDATA/derived/vessel_days/_build.json" ] &&
    ! cmp -s "$SDATA/derived/vessel_days/_build.json" "$DATA/derived/vessel_days/_build.json"; then
    echo "note: places.json has changed since the older days of vessel_days were built" \
        "($(cat "$DATA/derived/vessel_days/_build.json")), so new and old days list different places"
fi

if [ -e "$PENDING" ]; then
    ok=1
    echo "=== $(date +%T) vessel_static"
    in_harness "$P7_PROJECT_DIR" build_vessel_days.py --static || ok=0
    restart p7-harness http://127.0.0.1:8110/health || ok=0
    if [ -z "${NO_DERIVE:-}" ]; then
        echo "=== $(date +%T) derive"
        if AIS_DATA_DIR="$DATA" "$AIS" derive --no-build; then
            restart p7-map http://127.0.0.1:8200/health || ok=0
        else
            ok=0
        fi
    else
        ok=0
    fi
    if [ $ok = 1 ]; then
        rm -f "$PENDING"
    else
        echo "not everything after the new days ran: the next run does these steps again"
        [ -n "${NO_DERIVE:-}${NO_RESTART:-}" ] || failed=$((failed + 1))
    fi
fi
echo "=== $(date +%T) done"
[ "$failed" -eq 0 ]
