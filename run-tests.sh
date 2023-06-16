source venv/bin/activate
rm -r build
rm -r dist
pip uninstall decode-mcd -y
pip install .
python3 -m unittest -v
