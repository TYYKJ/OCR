BIN_FILE=$(readlink -f "$0")
PROJ_BIN=$(dirname "$BIN_FILE")
PROJ_HOME=$(dirname $PROJ_BIN)

export PYTHONPATH="${PYTHONPATH}:${PROJ_HOME}"
python $(dirname $(readlink -f "$0"))/flaskBackend.py