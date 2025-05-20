
---

```markdown
# Rehab Training

This project implements a rehabilitation movement classification pipeline using deep learning models (GRU and LSTM). It includes raw data preprocessing, dataset generation via sliding window, and model training.

## 📁 Project Structure

```

rehab\_training/
├── data\_upload.py           # Preprocess and merge raw sensor data (32 → 16 channels)
├── sliding\_window\.py        # Perform sliding window on preprocessed data
├── data/
│   ├── dataset\_gen.py       # Generate X.npy and y.npy from windowed data
│   ├── X.npy                # Final input features
│   └── y.npy                # Corresponding labels
├── GRU.py                   # Train and evaluate GRU-based model
├── LSTM.py                  # Train and evaluate LSTM-based model
└── models/
└── best\_\*.pth           # Trained model files (GRU/LSTM)

```

## 🚀 Workflow

1. **Download Raw Data**  
   Download and place the raw data in the appropriate local directory.

2. **Data Preprocessing**  
   Run `data_upload.py` to preprocess the raw data by merging the 32-channel sensor data into 16 channels.

3. **Sliding Window Operation**  
   Run `sliding_window.py` to apply sliding windows to the preprocessed data.

4. **Dataset Generation**  
   Run `data/dataset_gen.py` to generate the training dataset and labels:
   - `X.npy`: input features.
   - `y.npy`: corresponding class labels.
   These files will be saved in the `data/` directory.

5. **Model Training**  
   - Use `GRU.py` to train a GRU-based classification model.
   - Use `LSTM.py` to train an LSTM-based classification model.  
   The best-performing models will be saved in the `models/` folder.

## 📦 Output Files

- `data/X.npy`: Input sample features for model training.
- `data/y.npy`: Class labels.
- `models/best_GRU_model.pth` / `best_LSTM_model.pth`: Trained model checkpoints.

## 🖥️ Environment Requirements

- Python ≥ 3.7
- PyTorch ≥ 1.10
- NumPy
- scikit-learn

## 📌 Notes

- Make sure to run each script in the order defined above.
- Adjust paths in the scripts according to your local directory structure if needed.

## 📞 Contact

For questions or support, please contact the project maintainer.

```

---

