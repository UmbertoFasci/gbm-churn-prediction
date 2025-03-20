# Introduction
Procedure outlining the prediction of customer churn as a binary target. Models such as Random Forest Classifier, LightGBM, XGBoost, and CatBoost are tuned and trained on the dataset in order to build an optimal customer churn prediction model. Included in this documentation is an exploratory data analysis which takes an in depth look at the various features across the various datasets.

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

- contract.csv — contract information
- personal.csv — the client's personal data
- internet.csv — information about Internet services
- phone.csv — information about telephone services

In each file, the column customerID contains a unique code assigned to each client.

The contract information is valid as of February 1, 2020.

# Analysis



