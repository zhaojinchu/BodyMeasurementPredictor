# Body Measurement Predictor

The Body Measurement Predictor is a machine learning pipeline and desktop application that estimates how a male client's body measurements may evolve after a change in weight. The project combines data preprocessing, a PyTorch regression model, and a Tkinter-based GUI that can be packaged as a standalone executable for offline use.

## Key Features
- **End-to-end workflow** covering preprocessing, training, inference, and user-facing prediction.
- **Synthetic data augmentation** that simulates multiple weight-change scenarios to improve model robustness.
- **Neural-network regression model** trained on scaled measurement deltas with special handling to limit unrealistic skeletal growth.
- **Interactive desktop tool** built with Tkinter for quick what-if analyses with adjustable target weights.
- **PyInstaller packaging** instructions for distributing the GUI as a single-file executable (too large to include in Git).

## Repository Structure

```
├── body_measurement_app3.py     # Tkinter GUI for entering measurements and showing predictions
├── infer.py                     # CLI script demonstrating inference with the trained network
├── preprocess.py                # Data preparation, scaling, and augmentation pipeline
├── train2.py / train3.py        # PyTorch training scripts (train3 adds skeletal constraints)
├── best_model.pth               # Saved neural-network weights from the latest training run
├── scaler_input.pkl             # Fitted StandardScaler for model inputs
├── scaler_output.pkl            # Fitted StandardScaler for model outputs (measurement deltas)
├── preprocessed_dataset.csv     # Augmented dataset produced by `preprocess.py`
├── merged_dataset.csv           # Raw source dataset (must be present before preprocessing)
├── requirements.txt             # Python dependencies for preprocessing, training, and the GUI
└── README.md                    # Project documentation (this file)
```

Historical experiments and alternative model definitions are stored in the `old_file_version/` directory for reference.

## Data Pipeline
1. **Source data**: `merged_dataset.csv` must contain at least the columns listed in `input_features` plus a `gender` column. Only rows where `gender == "male"` are used.
2. **Feature engineering**: `preprocess.py` generates a `new_weight` column to simulate weight fluctuations and computes percentage deltas for key measurements.
3. **Augmentation**: Multiple synthetic rows are created per record using 10%, 15%, and 20% weight gains to diversify the training signal.
4. **Scaling**: Separate `StandardScaler` instances are fitted for inputs and output deltas and persisted to `scaler_input.pkl` and `scaler_output.pkl`.
5. **Output**: The augmented dataset is saved as `preprocessed_dataset.csv` for downstream training scripts.

Run the preprocessing step whenever the raw dataset changes:

```bash
python preprocess.py
```

## Model Training
Two training scripts are provided:
- `train2.py` trains a baseline fully connected network on the scaled percentage deltas.
- `train3.py` extends the baseline with constrained skeletal feature updates and a custom weighted loss; this is the recommended entry point.

Both scripts load the scalers from preprocessing, split the dataset into train/test subsets, and train for 500 epochs using Adam with learning-rate scheduling. The weights with the lowest validation loss are saved to `best_model.pth`.

Train (or retrain) the recommended model with:

```bash
python train3.py
```

## Inference Workflows
### Command-line demonstration
`infer.py` illustrates how to prepare an input payload, run the trained network, reverse the output scaling, and constrain skeletal changes relative to weight deltas.

```bash
python infer.py
```

Update the `test_input` dictionary in the script to test different scenarios. The script prints a dictionary of predicted measurements in centimeters.

### Desktop application
`body_measurement_app3.py` provides a Tkinter GUI that mirrors the inference logic. Users enter current measurements, current weight, and target weight; the app displays projected measurements, automatically limiting ankle, wrist, and shoulder-to-crotch growth.

Launch the GUI with:

```bash
python body_measurement_app3.py
```

### Packaging as an executable
The Tkinter app can be packaged with PyInstaller for distribution. Because the resulting executable exceeds typical repository limits, it is not checked into Git.

```bash
pyinstaller --onefile --windowed \
  --hidden-import=sklearn \
  --hidden-import=sklearn.utils._typedefs \
  --hidden-import=sklearn.utils._heap \
  --add-data "scaler_input.pkl;." \
  --add-data "scaler_output.pkl;." \
  --add-data "best_model.pth;." \
  body_measurement_app3.py
```

The generated executable is placed in `dist/`. Copy the scaler `.pkl` files and `best_model.pth` if your packaging workflow does not embed them automatically.

## Environment Setup
1. Install Python 3.11 or later.
2. (Optional) Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Input and Output Features
- **Inputs** (`input_features`): ankle, arm-length, bicep, calf, chest, forearm, height, hip, leg-length, shoulder-breadth, shoulder-to-crotch, thigh, waist, wrist, height_cm, weight_kg, new_weight.
- **Predicted outputs**: percentage deltas for ankle, bicep, calf, chest, forearm, hip, shoulder-breadth, shoulder-to-crotch, thigh, waist, and wrist. Post-processing converts these back to centimeter estimates with skeletal constraints.

## Troubleshooting
- Ensure `merged_dataset.csv` is present before running `preprocess.py`.
- Delete stale scaler or model files if you rerun preprocessing from scratch; new scalers must match the dataset used for training and inference.
- If PyInstaller packaging fails on Windows, verify that `pywin32-ctypes` is installed (included in `requirements.txt`).

## Next Steps
Potential improvements include support for additional genders, richer augmentation strategies, quantifying prediction uncertainty, and improving the UI to visualize trajectories over multiple weight goals.

