Packaged exe is too large to push to git.

Install python dependencies:
pip install -r requirements.txt

Package file using:
pyinstaller --onefile --windowed --hidden-import=sklearn --hidden-import=sklearn.utils._typedefs --hidden-import=sklearn.utils._heap --add-data "scaler_input.pkl;." --add-data "scaler_output.pkl;." --add-data "best_model.pth;." body_measurement_app3.py

