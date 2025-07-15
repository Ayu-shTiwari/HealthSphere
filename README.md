# HealthSphere: Disease Prediction and Recommendation System

HealthSphere is a machine learning-powered application for disease prediction based on user symptoms. It provides not only the most probable diseases but also detailed recommendations, including disease descriptions, home precautions, dietary suggestions, and recommended medications. The project leverages multiple machine learning models and a rich set of healthcare datasets.

## Features

- Predicts the top 5 most probable diseases based on user-input symptoms
- Provides disease descriptions, home care precautions, dietary advice, and medication recommendations
- Compares and visualizes the performance of various machine learning classifiers
- Interactive data exploration and visualization using Jupyter Notebook
- Modular and extensible codebase for easy experimentation

## Project Structure

```
├── app_streamlit_updated.py         # (Optional) Streamlit app for web interface
├── miracle.ipynb                   # Main Jupyter Notebook for data analysis and modeling
├── datasets/                       # Folder containing all CSV datasets
│   ├── dataset.csv
│   ├── description.csv
│   ├── diets.csv
│   ├── dis_sym_dataset_comb.csv
│   ├── dis_sym_dataset_norm.csv
│   ├── medications.csv
│   ├── precautions_df.csv
│   ├── Symptom-severity.csv
│   └── symtoms_df.csv
├── models/                         # Folder containing trained model .pkl files
│   ├── ExtraTrees.pkl
│   ├── GradientBoost.pkl
│   ├── KNeighbors.pkl
│   ├── LogisticRegression.pkl
│   ├── MultiLayer Perception.pkl
│   ├── MultinomialNB.pkl
│   ├── Random Forest.pkl
│   ├── SVC.pkl
│   ├── SVM .pkl
│   └── XGBoost.pkl
```

## Datasets

- **dis_sym_dataset_comb.csv**: Combined dataset of diseases and symptoms
- **dis_sym_dataset_norm.csv**: Normalized version of the above
- **description.csv**: Disease descriptions
- **precautions_df.csv**: Home care precautions for diseases
- **diets.csv**: Dietary recommendations for diseases
- **medications.csv**: Medication recommendations for diseases
- **Symptom-severity.csv**: Symptom severity scores
- **symtoms_df.csv**: Additional symptom data

## How It Works

1. **Data Loading & Preprocessing**: Loads and cleans datasets, handles missing values, and encodes labels.
2. **Model Training**: Trains multiple classifiers (SVM, Random Forest, KNN, etc.) and saves them as `.pkl` files.
3. **Evaluation**: Compares models using accuracy, cross-validation, and confusion matrices.
4. **Prediction**: Accepts user symptoms, creates a feature vector, and predicts the top 5 diseases using the best model.
5. **Recommendation**: For each predicted disease, fetches description, precautions, diet, and medication from the datasets.
6. **Visualization**: Plots model performance and prediction probabilities.

## Usage

### 1. Jupyter Notebook
- Open `miracle.ipynb` in Jupyter or VS Code.
- Run the cells sequentially to:
  - Explore data
  - Train and evaluate models
  - Make predictions and view recommendations

### 2. Streamlit App (Optional)
- If `app_streamlit_updated.py` is present, run:
  ```
  streamlit run app_streamlit_updated.py
  ```
- Interact with the web interface for predictions.

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- wordcloud
- xgboost
- lightgbm
- (Optional) streamlit

Install dependencies with:
```
pip install -r requirements.txt
```

## Notes
- Ensure all dataset CSV files are present in the `datasets/` folder.
- Trained model files (`.pkl`) should be in the `models/` folder.
- For best results, keep disease names consistent (case, spacing) across all datasets.

## Troubleshooting
- If lists like precautions, diet, or medication are empty, check for exact disease name matches between predictions and dataset entries.
- If a model fails to load, retrain and re-save the model using the notebook.

## License
This project is for educational and research purposes only. Not for clinical use.

## Acknowledgements
- Inspired by open healthcare datasets and machine learning research.
- Built with Python, scikit-learn, and the open-source community.
