# Project: Crystallum

![image](assets/images/credit-card-stock.png)

#### Team Members

• Hwa Hara, Fannie Polcari, Dantrell Person, Huda Alghazouli, Kafayat Lawal, Angele Gueupi, William Mills

## Project Summary:

We are a credit card company named Crystallum looking to improve our credit lending processes.

## Problem Statement:

Analyze current customer metrics to assist credit team and leadership in making data backed decsions on what credit ranges to give newly approved customers, and determine which customers are likely to pay off their credit balance.

## What Business Impact does this have?/ What is the Business Value?

Our analysis will increase Crystallum profits by imroving vetting processes on new credit customers so that more top applicants with a high likelihood of paying back their debt get high credit limits. This also reduces risk by not extending higher credit limits to more risky customers.

## Stakeholders:

Leadership team, credit team, debt collection team, sales team

## Scope (Tools used):

Python, Pandas, Matplotlib, Linear regression, Classification, Logistic Regression, Random Forest Classifier, Tableau, SQL

## Tasks:

1. Pick technology pieces to support the project. (3 or more)
2. Present how we trained a then used machine learning in this real-world scenario that can lead to a Tableau.
3. In Tableau further exampling the use of machine learning by showcasing a few individuals and how that data can be analyzed and relied on for making business decisions for Crystallum

## Data Source:

We selected the following files from the Data Set:
The first dataset consists of 10,127 customers mentioning their age, salary, marital_status, credit card limit, credit card category, etc. There are nearly 18 features.

https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers

This second dataset consists of 777,715 customers mentioning: Gender,car owner, property owner, Number of children, Annual income, Income category, Education level, Marital status,Way of living, Birthday, Start date of employment, Is there a mobile phone, Is there a work phone, Is there a phone, Is there an email, Occupation, Family size.

https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction

## Outline details

First data set reviewed with tableau for our 10.127 customers as starting point.
![image](https://user-images.githubusercontent.com/106934375/204407612-f4de9a85-6889-44b1-9a30-6ed701598686.png)

## Machine Learning Models:

# Linear Regression Model:
* Used https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers to predict the credit limit for our potential customers. 








# Logistic Regression model:
* Used https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction to predict whether the customers who receive credit from us  will pay us back.
* We looked at the status of payments and reworked the values in the ‘STATUS’ column so it can only hold 0 or 1.
* Created a Scatter plot to show status with every other column.
* Used “X dummies” to convert all values to numbers (encoding).
* We've done three ways to resampling the values in the 'STATUS' column.

## STATUS Column values:
* 0: 1-29 days past due.
* 1: 30-59 days past due. 
* 2: 60-89 days overdue.
* 3: 90-119 days overdue. 
* 4: 120-149 days overdue. 
* 5: Overdue or bad debts, write-offs for more than 150 days.
* C: paid off that month.
* X: No loan for the month.

**Trial #1:**
``` def new_status_trial1(sampleTrial1):
    if sampleTrial1=='C' or sampleTrial1=='X':
        return 1
    else:
        return 0
        ```
