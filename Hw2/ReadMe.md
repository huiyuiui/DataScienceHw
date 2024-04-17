## DataScience Hw2

### Goal
Give a dataset with `imbalanced` class label, train a `supervised multi-class classification` model to predict the labels of testing data.

### Platform
- Kaggle
- Private Rank: `3/108`
- Private Score: 0.82988

### Preprocessing
- Missing value handled by `SimpleImputer`
- Normalization by `StandardScaler`
- Categorical encoding by `OneHotEncoder`
- Features combination by `PolynomialFeatures`

### Training
- Models: `RandomForest`, `XGBoost`, `CatBoost`
- Features select on `feature_importances_`
- Majority vote to predict test label


### More Detail
Please refer to [Spec](./NTHU-2024DS-hw2-Classification.pdf).