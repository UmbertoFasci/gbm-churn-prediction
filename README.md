# Introduction
Procedure outlining the prediction of customer churn as a binary target. Models such as **Random Forest Classifier**, **LightGBM**, **XGBoost**, and **CatBoost** are tuned and trained on the dataset in order to build an optimal customer churn prediction model. Included in this documentation is an exploratory data analysis which takes an in depth look at the various features across the various datasets.

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

## Data Processing

As an initial handling of the data, `contract_df` only requires a handling of the datetime types, and feature engineering in the form of creating the `ContractDuration`. This can be done under the assumption that the data was collected on **Feburary 1, 2020**. With this constant, we can calculate the contract duration of non-churned customers.

### Contract Data

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


