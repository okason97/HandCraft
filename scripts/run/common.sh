# Sourced by the run scripts. Sets DATA_DIR and SAVE_DIR and removes their options from the arguments.
# --data-root <dir>: directory with the datasets (default: $HANDCRAFT_DATA)
# --save-root <dir>: directory for checkpoints and generated datasets (default: $HANDCRAFT_SAVE)
DATA_DIR=${HANDCRAFT_DATA:-}
SAVE_DIR=${HANDCRAFT_SAVE:-}

args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --data-root|--save-root)
            if [ $# -lt 2 ]; then echo "$1 needs a directory" >&2; exit 1; fi
            if [ "$1" = "--data-root" ]; then DATA_DIR=$2; else SAVE_DIR=$2; fi
            shift 2;;
        --data-root=*) DATA_DIR=${1#*=}; shift;;
        --save-root=*) SAVE_DIR=${1#*=}; shift;;
        *) args+=("$1"); shift;;
    esac
done
set -- "${args[@]}"

if [ -z "$DATA_DIR" ]; then echo "Set HANDCRAFT_DATA or pass --data-root <dir>: the directory with the datasets" >&2; exit 1; fi
if [ -z "$SAVE_DIR" ]; then echo "Set HANDCRAFT_SAVE or pass --save-root <dir>: the directory for checkpoints and generated datasets" >&2; exit 1; fi
