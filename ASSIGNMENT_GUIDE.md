# Data Science Assignments Guide

## Important Note on Visualizations

**All visualizations should display directly in the notebook cell output.** Do NOT save images to the file system. Use `plt.show()` instead of `plt.savefig()`.

```python
# CORRECT - Display in cell output
plt.show()

# INCORRECT - Do not save to files
# plt.savefig('image.png')
```

---

## Assignment 1: Basic Statistics & Data Preprocessing
**File:** `01_Basic_Stats_1.ipynb`
**Dataset:** `Datasets/sales_data_with_discounts.csv`

### What to Do:
1. **Descriptive Statistics**
   - Calculate mean, median, mode for numerical columns
   - Calculate standard deviation and variance
   - Generate summary statistics using `.describe()`

2. **Data Visualization**
   - Create histograms for numerical columns to show distributions
   - Create boxplots to identify outliers and understand data spread
   - Create bar charts for categorical variables

3. **Outlier Detection**
   - Use boxplots to visually identify outliers
   - Apply IQR method to detect outliers programmatically

4. **Data Preprocessing**
   - Apply Z-score standardization to numerical features
   - Verify standardization (mean should be ~0, std should be ~1)
   - Apply one-hot encoding to categorical variables

---

## Assignment 2: Confidence Intervals
**File:** `02_Basic_Stats_2.ipynb`
**Dataset:** Print-head durability data (provided in notebook)

### What to Do:
1. **Sample Statistics**
   - Calculate sample mean and standard deviation
   - Understand the sample size (n=15)

2. **T-Distribution Confidence Interval**
   - Build 99% confidence interval using t-distribution (for small samples)
   - Use `scipy.stats.t.interval()`

3. **Z-Distribution Confidence Interval**
   - Build 99% confidence interval using z-distribution (when population std is known)
   - Use `scipy.stats.norm.interval()`

4. **Comparison**
   - Compare both methods and explain why t-distribution gives wider interval

---

## Assignment 3: Basics of Python
**File:** `03_Basics_of_Python.ipynb`

### What to Do:
1. **Prime Number Checker**
   - Create a function to check if a number is prime
   - Test with multiple numbers

2. **Multiplication Quiz**
   - Generate random numbers for a multiplication quiz
   - Use loops and user input

3. **Loop Exercise**
   - Print squares of even numbers between 100 and 200

4. **Word Counter**
   - Use dictionaries to count word frequency in text

5. **Palindrome Checker**
   - Create function to check if a string is palindrome
   - Use string manipulation and reversal

---

## Assignment 4: Hypothesis Testing
**File:** `04_Hypothesis_Testing.ipynb`

### What to Do:
1. **One-Sample Z-Test** (Restaurant franchise operating costs)
   - State null hypothesis (H0: mu = 4000)
   - State alternative hypothesis (H1: mu > 4000)
   - Calculate Z-statistic
   - Find p-value and compare with significance level
   - Make conclusion

2. **Chi-Square Test** (Device type vs Customer satisfaction)
   - Create contingency table
   - State hypotheses (H0: variables are independent)
   - Calculate chi-square statistic
   - Compare with critical value
   - Make conclusion about association

---

## Assignment 5: EDA - Cardiotocographic Data
**File:** `05_EDA_1.ipynb`
**Dataset:** `Datasets/Cardiotocographic.csv`

### What to Do:
1. **Data Exploration**
   - Load data and check shape, dtypes
   - Generate statistical summary
   - Check for missing values

2. **Distribution Analysis**
   - Create histograms for all features
   - Create boxplots to understand spread

3. **Correlation Analysis**
   - Create correlation matrix
   - Visualize with heatmap
   - Identify strongly correlated features

4. **Target Variable Analysis**
   - Create violin plots showing feature distributions by fetal state
   - Identify which features best indicate fetal health

5. **Outlier Detection**
   - Apply IQR method
   - Document outliers found in each feature

---

## Assignment 6: Multiple Linear Regression
**File:** `06_MLR.ipynb`
**Dataset:** `Datasets/ToyotaCorolla - MLR.csv`

### What to Do:
1. **Data Preparation**
   - Load dataset
   - Select relevant features for price prediction
   - Handle missing values if any

2. **Model Building**
   - Split data into train/test sets
   - Fit multiple linear regression model

3. **Model Evaluation**
   - Calculate R-squared
   - Extract and interpret coefficients
   - Perform residual analysis

4. **Visualization**
   - Create actual vs predicted scatter plot
   - Plot residuals

---

## Assignment 7: Logistic Regression
**File:** `07_Logistic_Regression.ipynb`
**Dataset:** `Datasets/Titanic_train.csv`, `Datasets/Titanic_test.csv`

### What to Do:
1. **Data Preprocessing**
   - Handle missing values (Age with median, Embarked with mode)
   - Encode categorical variables (Sex, Embarked)
   - Select features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

2. **Model Building**
   - Split data
   - Fit logistic regression model

3. **Model Evaluation**
   - Calculate Accuracy, Precision, Recall, F1-Score
   - Plot ROC curve and calculate AUC
   - Create confusion matrix

4. **Interpretation**
   - Identify most important features
   - Explain survival patterns by gender and class

---

## Assignment 8: Clustering
**File:** `08_Clustering.ipynb`
**Dataset:** `Datasets/EastWestAirlines.xlsx`

### What to Do:
1. **K-Means Clustering**
   - Apply elbow method to find optimal K
   - Fit K-Means with optimal clusters
   - Calculate silhouette score

2. **Hierarchical Clustering**
   - Create dendrogram
   - Fit AgglomerativeClustering

3. **DBSCAN**
   - Apply DBSCAN algorithm
   - Identify noise points (labeled -1)

4. **Comparison**
   - Create 2D visualization comparing all three algorithms
   - Analyze cluster characteristics using groupby().mean()

---

## Assignment 9: Principal Component Analysis
**File:** `09_PCA.ipynb`
**Dataset:** `Datasets/wine.csv`

### What to Do:
1. **Apply PCA**
   - Standardize features first
   - Fit PCA on all components

2. **Variance Analysis**
   - Create scree plot (explained variance per component)
   - Create cumulative variance plot
   - Mark 95% variance threshold

3. **Dimensionality Reduction**
   - Reduce to 2 components for visualization
   - Create scatter plot of transformed data

4. **Clustering on PCA Data**
   - Apply K-Means on PCA-transformed data
   - Visualize clusters in 2D

---

## Assignment 10: Association Rules
**File:** `10_Association_Rules.ipynb`
**Dataset:** `Datasets/Online retail.xlsx`

### What to Do:
1. **Data Cleaning**
   - Remove cancelled transactions (InvoiceNo starts with 'C')
   - Remove rows with missing descriptions

2. **Transaction Matrix**
   - Create basket encoding (binary matrix of items per transaction)

3. **Apriori Algorithm**
   - Set minimum support (e.g., 2%)
   - Generate frequent itemsets

4. **Association Rules**
   - Generate rules with lift metric
   - Filter rules: lift >= 2, confidence >= 0.5
   - Identify top rules

5. **Visualization**
   - Create support vs confidence scatter plot

---

## Assignment 11: Recommendation System
**File:** `11_Recommendation_System.ipynb`
**Dataset:** `Datasets/anime.csv`

### What to Do:
1. **Content-Based Filtering**
   - Apply TF-IDF on genre column
   - Compute cosine similarity matrix

2. **Recommendation Function**
   - Create function to get recommendations for given anime
   - Return top N similar anime

3. **Popularity-Based Baseline**
   - Identify top-rated anime
   - Create simple popularity-based recommendations

---

## Assignment 12: Advanced EDA & Feature Engineering
**File:** `12_EDA_2.ipynb`
**Dataset:** `Datasets/adult_with_headers.csv`

### What to Do:
1. **Scaling Techniques**
   - Apply Standard Scaling (verify mean=0, std=1)
   - Apply Min-Max Scaling (verify range [0,1])

2. **Encoding**
   - One-hot encode low-cardinality features (<5 categories)
   - Label encode high-cardinality features (>=5 categories)

3. **Feature Engineering**
   - Create `total_capital` = capital_gain - capital_loss
   - Create `age_group` (Young/Middle/Senior/Elder)
   - Apply log transformation to skewed features

4. **Feature Selection**
   - Apply Isolation Forest for outlier detection
   - Analyze correlation matrix
   - Calculate Predictive Power Score (PPS)

---

## Assignment 13: Decision Tree
**File:** `13_Decision_Tree.ipynb`
**Dataset:** `Datasets/heart_disease.xlsx`

### What to Do:
1. **Data Preparation**
   - Load and preprocess data
   - Train-test split (80-20)

2. **Model Building**
   - Train default decision tree
   - Evaluate performance

3. **Hyperparameter Tuning**
   - Test max_depth: [3, 5, 7, 10, None]
   - Test min_samples_split: [2, 5, 10]
   - Test criterion: [gini, entropy]

4. **Evaluation**
   - Calculate Accuracy, Precision, Recall, F1, ROC-AUC
   - Visualize tree (first 3 levels)
   - Extract and plot feature importance

---

## Assignment 14: Random Forest
**File:** `14_Random_Forest.ipynb`
**Dataset:** `Datasets/glass.xlsx`

### What to Do:
1. **Decision Tree Baseline**
   - Train decision tree classifier
   - Record accuracy

2. **Random Forest**
   - Train Random Forest classifier
   - Record accuracy

3. **Comparison**
   - Calculate improvement percentage
   - Generate classification report

4. **Feature Importance**
   - Extract feature importance from ensemble
   - Visualize top features

---

## Assignment 15: XGBoost & LightGBM
**File:** `15_XGBM_LGBM.ipynb`
**Dataset:** `Datasets/Titanic_train.csv`

### What to Do:
1. **XGBoost Model**
   - Set hyperparameters: n_estimators=100, max_depth=5, learning_rate=0.1
   - Train and evaluate

2. **LightGBM Model**
   - Use same hyperparameters
   - Train and evaluate

3. **Comparison**
   - Compare accuracy of both models
   - Plot feature importance for both
   - Create bar chart comparing performance

---

## Assignment 16: K-Nearest Neighbors
**File:** `16_KNN.ipynb`
**Dataset:** `Datasets/Zoo.csv`

### What to Do:
1. **Optimal K Selection**
   - Test K values from 1 to 20
   - Use elbow method to find optimal K

2. **Distance Metrics**
   - Compare Euclidean, Manhattan, Minkowski
   - Select best metric

3. **Model Evaluation**
   - Train final model with best K and metric
   - Generate confusion matrix (7x7 for animal types)

4. **Visualization**
   - Create decision boundary plot in 2D
   - Use milk and feathers features

---

## Assignment 17: Support Vector Machine
**File:** `17_SVM.ipynb`
**Dataset:** `Datasets/mushroom.csv`

### What to Do:
1. **Data Preprocessing**
   - Label encode all categorical features (20+ columns)
   - Apply StandardScaler (critical for SVM)

2. **Kernel Comparison**
   - Test Linear, RBF, Polynomial, Sigmoid kernels
   - Compare performance

3. **Hyperparameter Tuning**
   - Test C values: [0.1, 1, 10, 100]
   - Test gamma values: ['scale', 'auto', 0.1, 1]

4. **Final Model**
   - Train with best parameters
   - Generate classification report

---

## Assignment 18: Neural Networks
**File:** `18_Neural_Networks.ipynb`
**Dataset:** `Datasets/Alphabets_data.csv`

### What to Do:
1. **Model Architecture**
   - Test MLPClassifier with hidden layers: (128, 64, 32)
   - Use ReLU activation
   - Use Adam solver

2. **Architecture Comparison**
   - Compare architectures: (50,), (100,), (100, 50), (128, 64, 32)
   - Record accuracy for each

3. **Training Analysis**
   - Plot training loss curve
   - Calculate final accuracy metrics

---

## Assignment 19: Naive Bayes & Text Mining
**File:** `19_Naive_Bayes_Text_Mining.ipynb`
**Dataset:** `Datasets/blogs.csv`

### What to Do:
1. **Text Preprocessing**
   - Convert to lowercase
   - Remove emails and URLs
   - Remove punctuation and numbers
   - Strip extra whitespace

2. **Tokenization & Stopword Removal**
   - Split text into words
   - Remove NLTK English stopwords

3. **Feature Extraction**
   - Apply TF-IDF vectorization (max 5000 features)

4. **Classification**
   - Train Multinomial Naive Bayes
   - Evaluate performance

5. **Sentiment Analysis**
   - Use TextBlob for polarity scoring
   - Classify as Positive/Neutral/Negative
   - Visualize sentiment distribution by category

---

## Assignment 20: Time Series Analysis
**File:** `20_Time_Series.ipynb`
**Dataset:** `Datasets/exchange_rate.csv`

### What to Do:
1. **Time Series Visualization**
   - Plot time series data
   - Identify date range

2. **Decomposition**
   - Apply seasonal decomposition
   - Plot: Observed, Trend, Seasonal, Residual components

3. **Stationarity Testing**
   - Perform ADF (Augmented Dickey-Fuller) test
   - Determine if differencing is needed

4. **ACF/PACF Analysis**
   - Generate ACF and PACF plots
   - Identify appropriate lag values

5. **ARIMA Modeling**
   - Fit ARIMA model (order 1,1,1)
   - Generate forecast
   - Visualize forecast vs actual
   - Extract model summary statistics

---

## Datasets Reference

| Dataset | Assignment | Purpose |
|---------|------------|---------|
| sales_data_with_discounts.csv | 01 | Sales analysis & preprocessing |
| Cardiotocographic.csv | 05 | Fetal health EDA |
| ToyotaCorolla - MLR.csv | 06 | Car price regression |
| Titanic_train.csv, Titanic_test.csv | 07, 15 | Survival classification |
| EastWestAirlines.xlsx | 08 | Customer clustering |
| wine.csv | 09 | PCA analysis |
| Online retail.xlsx | 10 | Market basket analysis |
| anime.csv | 11 | Recommendations |
| adult_with_headers.csv | 12 | Income prediction EDA |
| heart_disease.xlsx | 13 | Heart disease decision tree |
| glass.xlsx | 14 | Glass classification |
| Zoo.csv | 16 | Animal KNN classification |
| mushroom.csv | 17 | Mushroom SVM classification |
| Alphabets_data.csv | 18 | Alphabet neural network |
| blogs.csv | 19 | Text classification |
| exchange_rate.csv | 20 | Time series forecasting |

---

## Skills Covered

- **Statistics:** Descriptive stats, confidence intervals, hypothesis testing
- **Data Processing:** Scaling, encoding, feature engineering, outlier detection
- **Visualization:** Histograms, boxplots, heatmaps, scatter plots
- **Regression:** Linear, Multiple Linear, Logistic
- **Classification:** Decision Tree, Random Forest, KNN, SVM, Naive Bayes, XGBoost, LightGBM, Neural Networks
- **Clustering:** K-Means, Hierarchical, DBSCAN
- **Dimensionality Reduction:** PCA
- **Advanced:** Association Rules, Recommendation Systems, NLP, Time Series
