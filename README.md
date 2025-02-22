# cifar10-data-mining-classification

## Project Overview
This project is part of the INFS7203 Data Mining course at The University of Queensland. The goal is to classify images from a CIFAR-10 derived dataset into one of 10 categories (0-9) using machine learning techniques. The dataset consists of 128 extracted features representing images, with a mix of numerical and categorical attributes.

The project applies data preprocessing, model selection, and hyperparameter tuning to improve classification performance. The final model is evaluated using macro-F1 score.

## Repository Structure

This repository is structured to allow users to run the code immediately without needing to move files manually.
```
data-mining-project/
│── README.md                  # Project Overview
│── requirements.txt            # List of dependencies for installation
│── src/
│   ├── main.py                 # Script for training the model
│── data/
│   ├── train.csv                # Original training dataset
│   ├── add_train.csv            # Additional training data
│   ├── test.csv                 # Test dataset without labels
│── results/         # Where the predicted labels were stored
│── reports/
│   ├── project_proposal.docx    # Initial project proposal
│   ├── final_report.docx        # Final report summarizing findings
```

## Setup
1. Clone the repository
`
git clone https://github.com/TsungTseTu122/DataMining-cifar10-classification.git
cd data-mining-project
`
2. Set Up a Virtual Environment (If needed to avoid conflict)
Windows:
`
python -m venv venv
venv\Scripts\activate
`
MacOS/Linux:
`
python3 -m venv venv
source venv/bin/activate
`
3. Install dependencies
Install the required Python packages:
`
pip install -r requirements.txt
`

### How to run
Training the Model (Using `main.py`)
1. Ensure the datasets (`train.csv`, `add_train.csv`, and `test.csv`) are placed inside the `data/` folder.
2. Run the training script from the `src/` folder:
`
python src/main.py
`
3. The script will do:
- Combine train.csv and add_train.csv into a single dataset.

- Preprocess numerical and categorical features.

- Train multiple classifiers (Decision Tree, Random Forest, KNN, Naïve Bayes).

- Use GridSearchCV for hyperparameter tuning.

- Perform cross-validation to evaluate accuracy and macro-F1 score.

- Save the final predictions in results/s4780187.csv.

## How it works
### Data processing
- Imputation: Missing values in numerical columns are replaced with the mean, while categorical columns use the most frequent value.

- Normalization: Applied Z-score normalization for numerical columns.

- Feature Engineering: Combined `train.csv` with `add_train.csv` for additional training samples.

### Classification models
The following classifiers were trained and evaluated:

- Decision Tree

- Random Forest

- k-Nearest Neighbors (KNN)

- Naïve Bayes

The best model was Random Forest, achieving:

Mean Accuracy: 0.978

Mean Macro-F1 Score: 0.972 (sometimes 0.971)

### Model evaluation
- Used 5-Fold Cross-Validation during hyperparameter tuning.

- Macro-F1 Score was used as the primary evaluation metric.

## Future improvement outside project requirement:
- Experiment with deep learning models instead of traditional machine learning models used in this project.

- Test additional feature selection techniques like PCA if more feature types were used.

- Generate synthetic data to streghten our training.
