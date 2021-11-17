work_path='/home/cat/PycharmProjects/torch-ocr'

export PYTHONPATH="${PYTHONPATH}:${work_path}"
python $(dirname $(readlink -f $0))/train.py
