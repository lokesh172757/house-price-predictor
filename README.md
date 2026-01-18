🏠 AI Real Estate Appraiser (Ames Housing)
![alt text](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)

![alt text](https://img.shields.io/badge/Python-3.9%2B-blue)

![alt text](https://img.shields.io/badge/Library-Scikit--Learn-orange)
Live Demo: https://ames-housing-ai.streamlit.app/
GitHub Repository: https://github.com/lokesh172757/house-price-predictor
📌 Project Overview

This project is an end-to-end machine learning solution designed to estimate residential property prices in Ames, Iowa. Unlike typical "black box" solutions, this project focuses on interpretable machine learning, employing a rigorous pipeline to handle the complex, high-dimensional nature of real estate data (80+ features).
The final model utilizes a Lasso Regression Pipeline that, surprisingly, outperformed state-of-the-art tree-based models (XGBoost/Random Forest) on this specific dataset due to the sparse nature of the data after encoding.

⚙️ The Challenge & Data Pipeline

The Ames Housing dataset is notorious for its complexity. It requires significant preprocessing before any model can learn effectively.

1. Data Cleaning & Outlier Removal
Outlier Detection: Visualized GrLivArea vs SalePrice and removed documented "Partial Sales" (houses > 4,000 sq ft sold at abnormally low prices) to prevent model skew.
Imputation: Implemented a dual-strategy for missing values:
Categorical (Pool, Alley, Fence): Filled with "None" (indicating the feature doesn't exist).
Numerical (LotFrontage): Filled with the Median value.

2. Feature Engineering

Target Transformation: The target variable (SalePrice) was highly right-skewed. Applied np.log1p (Log Transformation) to normalize the distribution, allowing linear models to perform effectively.
One-Hot Encoding: Converted categorical text data into numerical vectors. This expanded the dataset from 80 columns to ~300 columns, creating a high-dimensional sparse matrix.

3. Scaling

Applied StandardScaler to numerical features to ensure that variables like Area (thousands) didn't dominate variables like YearBuilt or encoded features (0/1).

🔬 Model Experimentation & Results

I conducted a rigorous comparison between Linear Models (which handle high-dimensional sparse data well) and Tree-Based Models (which typically dominate Kaggle competitions).
The metric used for evaluation was RMSE (Root Mean Squared Error) on the Log-Transformed target.

Model Algorithm	RMSE (Log Scale)	RMSE (Real Dollars)	Verdict
Decision Tree	0.21079	~$37,400	Severe Overfitting
Random Forest	0.14362	~$24,000	Good, but struggled with sparsity
XGBoost	0.13526	~$23,400	Complex, prone to overfitting here
Ridge Regression	0.12270	~$20,000	Very Strong
Lasso Regression	0.11698	~$19,300	🏆 CHAMPION

🧠 The Conclusion
Why did Lasso win?
While XGBoost is usually superior, in this specific case, One-Hot Encoding created a dataset with ~300 features, many of which were noise. Lasso (L1 Regularization) excels here because it forces the coefficients of weak features to become exactly Zero, effectively performing automatic Feature Selection. XGBoost struggled to find the signal amidst the sparse noise of the encoded categories.

Final Model Performance:
R2 Score (Test): 91.88%

Generalization: The gap between Train and Test accuracy was only ~2.1%, indicating zero overfitting.

🛠️ Tech Stack & Tools
Python: Core logic.
Scikit-Learn: Pipeline, GridSearchCV, ColumnTransformer, StandardScaler, OneHotEncoder.
XGBoost: For benchmarking against linear models.
Joblib: For serializing the model pipeline and default value dictionary.
Streamlit: For building the interactive web interface.
Pandas/NumPy: Data manipulation and vector math (log1p, expm1).

💻 How to Run Locally
Clone the repository:
git clone https://github.com/YOUR_USERNAME/house-price-predictor.git

Navigate to the folder:
cd house-price-predictor

Install dependencies:
pip install -r requirements.txt

Run the App:
streamlit run app.py

👨‍💻 Author
Lokesh Singh
[LinkedIn Profile](www.linkedin.com/in/lokesh-singh2588)
[GitHub Profile](https://github.com/lokesh172757/house-price-predictor)