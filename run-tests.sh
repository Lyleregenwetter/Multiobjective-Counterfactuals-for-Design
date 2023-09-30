set -x
source venv/bin/activate
rm -r build dist
pip uninstall decode-mcd -y
pip install .
python3 -m unittest -v
