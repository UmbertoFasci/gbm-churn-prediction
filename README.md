# Optimized Customer Churn Prediction with Gradient Boosting and Optuna Hyperparameter Tuning

<img src="https://github.com/UmbertoFasci/gbm-churn-prediction/blob/main/assets/customer_churn.png" alt="customer churn hero" style="width:100%;"/>

Procedure outlining the prediction of customer churn as a binary target. **Random Forest Classifier**, **LightGBM**, **XGBoost**, and **CatBoost** models are tuned and trained on the dataset in order to build an optimal customer churn prediction model. Included in this documentation is an exploratory data analysis which takes an in depth look at the various features across the various datasets, and a hyperparameter tuning procedure with the Optuna library.

## Prompt
A telecom operator would like to be able to forecast their churn of clients. If it's discovered that a user is planning to leave, they will be offered promotional codes and special plan options. Their marketing team has collected some of their clientele's personal data, including information about their plans and contracts.

Two types of services:

- Landline communication. The telephone can be connected to several lines simultaneously.
- Internet. The network can be set up via a telephone line (DSL, digital subscriber line) or through a fiber optic cable.

Some other services the company provides include:

- Internet security: antivirus software (DeviceProtection) and a malicious website blocker (OnlineSecurity)
- A dedicated technical support line (TechSupport)
- Cloud file storage and data backup (OnlineBackup)
- TV streaming (StreamingTV) and a movie directory (StreamingMovies)

The clients can choose either a monthly payment or sign a 1 or 2 year contract. They can use various payment methods and receive an electronic invoice after a transaction.

## Data Description

The data consists of files obtained from different sources:

- **contract.csv** — contract information
- **personal.csv** — the client's personal data
- **internet.csv** — information about Internet services
- **phone.csv** — information about telephone services

In each file, the column customerID contains a unique code assigned to each client.

The contract information is valid as of **February 1, 2020**.

# Analysis

After loading the data and performing a baseline assessment, the `contract` and `personal` datasets represet the complete customer base with a total of **7043** entries. While `internet` and `phone` customers, with **5517** and **6361** respective entries represent customers which have opted in these services.

Throughout the entire corpus of data, `customerID` represents the unique customer identifier in order to better link the entire customer profile.

The data processing which occurs in this section is solely for the purpose of completing the analysis and for initializing the modeling data. The data will be further processed for the modeling procedure in the following section.

## Contract

As an initial handling of the data, `contract_df` only requires a handling of the datetime types, and feature engineering in the form of creating the `ContractDuration`. This can be done under the assumption that the data was collected on **Feburary 1, 2020**. With this constant, we can calculate the contract duration of non-churned customers.

```python
date_collected = date(2020, 2, 1)

# preprocess contract data
def preprocess_contract_data(df):
    # Handle datetime
    df['BeginDate'] = pd.to_datetime(df['BeginDate'])
    df['EndDate'] = df['EndDate'].replace('No', pd.NaT)
    df['EndDate'] = pd.to_datetime(df['EndDate'])
    
    # Calculate contract duration in months for churned customers
    mask_churned = ~df['EndDate'].isna()
    df.loc[mask_churned, 'ContractDuration'] = (
        (df.loc[mask_churned, 'EndDate'] - df.loc[mask_churned, 'BeginDate']).dt.days / 30).round(1)
    
    # Calculate contract duration in months for non-churned customers (utilizing date_collected)
    date_collected_pd = pd.to_datetime(date_collected)  # Convert to pandas datetime
    df.loc[~mask_churned, 'ContractDuration'] = (
        (date_collected_pd - df.loc[~mask_churned, 'BeginDate']).dt.days / 30).round(1)
    
    # TotalCharges was presented as an object type, might have a non-numeric value causing this...
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # drop the coerced entries
    df = df.dropna(subset=['TotalCharges'])
    # Create churn indicator
    df['Churned'] = ~df['EndDate'].isna()
    
    return df
```

<img src="https://github.com/UmbertoFasci/gbm-churn-prediction/blob/main/assets/ctr_duration.png" alt="contract duration distribution" style="width:100%;"/>

The calculated feature, `ContractDuration`, expresses a distribution where customers have a contract duration between 1 and 73 months, with a large concentration of customers having signed a contract within the previous 10 months of the dataset generation. The largest group of customers can be presented as having a contract duration between 10 and 63 months (this can be further delineated), with a small group of customers having an active contract duration exceeding 63 months.

## Internet

The `internet` dataset is a simple catalog of the opted for (or not) internet services provided by the telecom operator. There is no need for any data preprocessing besides the generation of several new features through feature engineering.

```python
def preprocess_internet_data(df):
    # Map all Yes/No service indicators to 1/0
    service_columns = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    for col in service_columns:
        df[col] = df[col].map({
            'Yes': 1,
            'No': 0,
            'No internet service': 0
        })
    
    # Create a count of total services
    df['TotalInternetServices'] = df[service_columns].sum(axis=1)
    
    # Convert InternetService to categorical codes
    df['InternetService'] = pd.Categorical(df['InternetService'])
    
    return df
```

<img src="https://github.com/UmbertoFasci/gbm-churn-prediction/blob/main/assets/int_service_dis.png" alt="internet service distribution" style="width:100%;"/>

`TotalInternet` services represents the sum total of internet services opted-in for per customer across 6 different services. This distribution suggests a relatively uniform distribution favoring more opt-ins.

## Personal

Similar to the `internet` dataset, the `personal` presents a catalog of a customer's personal info presented in a categorical nature. Utilizing this kind of data, we must ensure it is applied in an ethical manner, and adhere to the telecom operator's risk management policies. Assuming the use of the data and the utilization of the data to determine customer targeting based on `gender`, `SeniorCitizen` status, `Partner` status, and `Dependents` status is completely allowed.

```python
def preprocess_personal_data(df):
    df = df.copy()
    
    # Convert Yes/No columns to binary
    binary_columns = ['Partner', 'Dependents']
    for col in binary_columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # Convert gender
    df['gender'] = pd.Categorical(df['gender'])
    
    return df
```
No new features can be inferred based on the provided personal data. `FamilySize` might be a potential feature worth looking into however the information necessary to calculate that is not available. In this case, generating binary maps for the several features included will suffice.

## Phone

No data processing for the phone dataset is needed at this point of the procedure.

## Feature Matrix

Merging all the previously processed datasets will allow for further analysis. It is important to note that since the datasets are not the same size there will be missing values introduced during the merge, and handling them appropriately is essential.

```python
def create_feature_matrix(contract_df, internet_df, personal_df, phone_df):
    # Initialize feature matrix with contract data
    feature_matrix = contract_df.copy()

    # Add service enrollment indicators before merging
    all_customers = feature_matrix['customerID']

    # Internet service enrollment
    feature_matrix['HasInternetService'] = feature_matrix['customerID'].isin(internet_df['customerID'])

    # Phone service enrollment
    feature_matrix['HasPhoneService'] = feature_matrix['customerID'].isin(phone_df['customerID'])
    
    # Add internet services
    feature_matrix = feature_matrix.merge(
        internet_df,
        on='customerID',
        how='left'
    )
    # Handle NA values for customers without internet service
    internet_columns = internet_df.columns.drop('customerID')
    
    # Add phone services
    feature_matrix = feature_matrix.merge(
        phone_df,
        on='customerID',
        how='left'
    )
    # Fill NA values for customers without phone service
    feature_matrix['MultipleLines'] = feature_matrix['MultipleLines'].fillna('No Service')
    
    # Add personal information
    feature_matrix = feature_matrix.merge(
        personal_df,
        on='customerID',
        how='left'
    )
    
    return feature_matrix

feature_matrix = create_feature_matrix(contract_df, internet_df, personal_df, phone_df)
```

In the case of handling the introduced missing values, the only two features that would be an effect of these artifacts are `internet` and `phone`. Another important aspect of this to note is that in order to be included in this dataset in the first place is to be a part of either of these two particular datasets as a subscriber to the appropriate plan.

With the feature matrix created, we can perform a more in depth analysis.

## Service Enrollment Analysis

<img src="https://github.com/UmbertoFasci/gbm-churn-prediction/blob/main/assets/svc_enrollment.png" alt="service enrollment" style="width:100%;"/>

Taking a look at the whole dataset, there are more customers enrolled in the `phone` plans, and more customers not opting in to the `internet` plan. To clarify, both plans are of a healthy ratio of opted-in customers. Where are the opted-out customers going? Since they are part of our dataset they must be enrolled in one of these two plans, so the 1520 not enrolled in an internet plan must only be enrolled in a phone plan, and vice-versa.

### Opt-in Pattern

<img src="https://github.com/UmbertoFasci/gbm-churn-prediction/blob/main/assets/svc_enrollment_comb.png" alt="service enrollment combination" style="width:100%;"/>

This reinforces what we determined earlier, the 1520 customers who have opted-out of the internet plan were enrolled in a **phone-only** service, and those who have opted out of the phone plan were enrolled in a **internet-only** service. There are no customer entries which are enrolled in `No Service`.

Most customers opt-in for the combination plan with internet and phone services, suggesting good converting practices targeting this particular plan.

### Contract Type by Service Group

<img src="https://github.com/UmbertoFasci/gbm-churn-prediction/blob/main/assets/svc_grp_chr.png" alt="service enrollment combination by contract" style="width:100%;"/>

Having a better understanding of the service group characteristics as it applies to contract length adoption allows us to imply a granular look at the potential relationships between the adoption length and service group selection. Across the board, the most common service group is `internet + phone` as presented in the previous analysis. The most common contract type in this service group is a `month-to-month` type which in most cases represents a default for contract selection for most subscribable services. The main outlier group in this case is the `phone only` service group with having the longest term contracts representing the majority of its base.

# Feature Processing

By defining each column into their respective feature type we can readily apply processing functions so that the data can be better utilized in the context of gradient boosting.

```python
# Define features
categorical_features = [
    'Type', 'PaperlessBilling', 'PaymentMethod',
    'InternetService', 'MultipleLines'
]
numerical_features = [
    'MonthlyCharges', 'TotalCharges', 'ContractDuration',
    'TotalInternetServices'
]
binary_features = [
    'HasInternetService', 'HasPhoneService', 
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Partner', 'Dependents', 'SeniorCitizen'
]
```
Here, all previously defined categorical features are processed accordingly to ensure that all values are of an expected cateogircal or string type and then encoded into specified numerical identifiers with the `LabelEncoder()`. This method is not scalabale to the extent of streaming or a large amount of categorical values, but works well in this application.

```python
# Preprocess the data
df = feature_matrix.copy()
label_encoders = {}
scaler = StandardScaler()

# Handle categorical features
for feature in categorical_features:
    if df[feature].dtype.name == 'category':
        categories = df[feature].cat.categories.tolist()
        df[feature] = df[feature].astype(str)
    else:
        df[feature] = df[feature].astype(str)
        categories = df[feature].unique().tolist()
    
    label_encoders[feature] = LabelEncoder()
    label_encoders[feature].fit(categories)
    df[feature] = df[feature].map(
        dict(zip(categories, label_encoders[feature].transform(categories)))
    )
```

The numerical features are handled by first imputing the median of each feature set to its missing values, and then scaling the values appropriatly utilizing the `StandardScalar()`. Defined binary features are handled by appropriatly typing them as boolean values, and the missing values in this case are considered to be negative. 

```python
# Handle numerical features
for feature in numerical_features:
    median_value = df[feature].median()
    df[feature] = df[feature].fillna(median_value)

df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Handle binary features
for feature in binary_features:
    if df[feature].dtype == bool:
        df[feature] = df[feature].astype(int)
    else:
        df[feature] = df[feature].fillna(0).astype(int)
```

Here the features and target variables are generated accordingly after all processing is completed, and the data is split with the specifications of 80-20 train-test split. `scale_pos_weight` is calculated to obtain a class ratio for model balancing during training for select models.

```python
# Prepare features and target
features = categorical_features + numerical_features + binary_features
X = df[features]
y = df['Churned'].astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate class weight for imbalanced data
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
```

# Model Tuning

In a GPU environment, all models were tuned with the **Optuna** library.

In addition to these paramter ranges, the `class_weight` parameter was utilized in order to handle the class imbalance of customers who have and have not churned.

**LightGBM** utilized a `balanced` class weight which automatically adjusts weights inversely proportional to the class frequencies, giving balanced importance to the minority class. **Random Forest** handles the class imbalances in the same way. **XGBoost** handles the class imbalance by the set ratio between the classes with the previously calculated `sclae_pos_weight` parameter. In a similar manner **CatBoost** implements the same ratio, however requires a list were index 0 represents the base class, and index 1 corresponds to the positive class.

# Modeling

```python
# Initialize models with pre-tuned parameters
models = {
    'LightGBM': LGBMClassifier(
        n_estimators=869,
        learning_rate=0.068647,
        num_leaves=67,
        max_depth=4,
        min_child_samples=38,
        subsample=0.697040,
        colsample_bytree=0.948016,
        class_weight='balanced',
        random_state=42
    ),
    'XGBoost': XGBClassifier(
        n_estimators=525,
        learning_rate=0.090247,
        max_depth=4,
        min_child_weight=2,
        subsample=0.880375,
        colsample_bytree=0.906231,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    ),
    'CatBoost': CatBoostClassifier(
        iterations=902,
        learning_rate=0.099470,
        depth=3,
        l2_leaf_reg=0.001837,
        bootstrap_type='Bernoulli',
        class_weights=[1, scale_pos_weight],
        random_seed=42,
        verbose=False
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=106,
        max_depth=20,
        min_samples_split=17,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42
    )
}
```

```python
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    
    results[name] = {
        'roc_auc': roc_auc,
        'accuracy': accuracy
    }
```