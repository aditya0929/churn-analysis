# End-to-End Churn Analysis Portfolio Project

## Introduction to Churn Analysis

In today's competitive business environment, retaining customers is crucial for long-term success. Churn analysis is a key technique used to understand and reduce customer attrition. By examining customer data to identify patterns and reasons behind customer departures, businesses can predict which customers are at risk of leaving and understand the factors driving their decisions. This knowledge allows companies to take proactive steps to improve customer satisfaction and loyalty.

## Data & Resources

- **Colours Used:** #4A44F2, #9B9FF2, #F2F2F2, #A0D1FF
- **data set used:** https://e3da6ab4-ff6e-4f55-bfa1-a8fb6979d99b.usrfiles.com/archives/e3da6a_13daacb2b66940e1906249e106e5984c.zip

## Target Audience

Although this project focuses on churn analysis for a telecom firm, the techniques and insights are applicable across various industries. From retail and finance to healthcare, any business that values customer retention can benefit from churn analysis. We will explore the methods, tools, and best practices for reducing churn and improving customer loyalty, transforming data into actionable insights for sustained success.

## Project Goals

- **Create an entire ETL process in a database and a Power BI dashboard to utilize the Customer Data and achieve the following goals:**
  - Visualize & Analyze Customer Data at various levels (Demographic, Geographic, Payment & Account Info, Services)
  - Study Churner Profile & Identify Areas for Implementing Marketing Campaigns
  - Identify a Method to Predict Future Churners

## Metrics Required

- Total Customers
- Total Churn & Churn Rate
- New Joiners

## Step 1: ETL Process in SQL Server

1. **Download SSMS**: [Download SQL Server Management Studio](https://learn.microsoft.com/en-us/sql/ssms/download-sql-server-management-studio-ssms?view=sql-server-ver16)
2. **Creating Database**:
    ```sql
    CREATE DATABASE db_Churn
    ```
3. **Import CSV into SQL Server Staging Table – Import Wizard**:
    - Task >> Import >> Flat file >> Browse CSV file
    - Add `customerId` as primary key and allow nulls for remaining columns
4. **Data Exploration – Check Distinct Values**:
    ```sql
    SELECT Gender, Count(Gender) as TotalCount, Count(Gender) * 1.0 / (Select Count(*) from stg_Churn)  as Percentage
    FROM stg_Churn
    GROUP BY Gender
    ```
5. **Data Exploration – Check Nulls**:
    ```sql
    SELECT 
        SUM(CASE WHEN Customer_ID IS NULL THEN 1 ELSE 0 END) AS Customer_ID_Null_Count,
        ...
        SUM(CASE WHEN Churn_Reason IS NULL THEN 1 ELSE 0 END) AS Churn_Reason_Null_Count
    FROM stg_Churn;
    ```
6. **Remove nulls and insert new data into Prod table**:
    ```sql
    SELECT 
        Customer_ID, Gender, Age, Married, State, Number_of_Referrals, Tenure_in_Months,
        ISNULL(Value_Deal, 'None') AS Value_Deal, Phone_Service, ISNULL(Multiple_Lines, 'No') As Multiple_Lines,
        Internet_Service, ISNULL(Internet_Type, 'None') AS Internet_Type, ISNULL(Online_Security, 'No') AS Online_Security,
        ISNULL(Online_Backup, 'No') AS Online_Backup, ISNULL(Device_Protection_Plan, 'No') AS Device_Protection_Plan,
        ISNULL(Premium_Support, 'No') AS Premium_Support, ISNULL(Streaming_TV, 'No') AS Streaming_TV,
        ISNULL(Streaming_Movies, 'No') AS Streaming_Movies, ISNULL(Streaming_Music, 'No') AS Streaming_Music,
        ISNULL(Unlimited_Data, 'No') AS Unlimited_Data, Contract, Paperless_Billing, Payment_Method,
        Monthly_Charge, Total_Charges, Total_Refunds, Total_Extra_Data_Charges, Total_Long_Distance_Charges,
        Total_Revenue, Customer_Status, ISNULL(Churn_Category, 'Others') AS Churn_Category, 
        ISNULL(Churn_Reason, 'Others') AS Churn_Reason
    INTO [db_Churn].[dbo].[prod_Churn]
    FROM [db_Churn].[dbo].[stg_Churn];
    ```
7. **Create View for Power BI**:
    ```sql
    CREATE VIEW vw_ChurnData AS
    SELECT * FROM prod_Churn WHERE Customer_Status IN ('Churned', 'Stayed')
    ```

## Step 2: Power BI Transform

1. **Add New Column in `prod_Churn`**:
    - `Churn Status = if [Customer_Status] = "Churned" then 1 else 0`
    - `Monthly Charge Range = if [Monthly_Charge] < 20 then "< 20" else if [Monthly_Charge] < 50 then "20-50" else if [Monthly_Charge] < 100 then "50-100" else "> 100"`
2. **Create New Table Reference for `mapping_AgeGrp`**:
    - `Age Group = if [Age] < 20 then "< 20" else if [Age] < 36 then "20 - 35" else if [Age] < 51 then "36 - 50" else "> 50"`
    - `AgeGrpSorting = if [Age Group] = "< 20" then 1 else if [Age Group] = "20 - 35" then 2 else if [Age Group] = "36 - 50" then 3 else 4`
3. **Create New Table Reference for `mapping_TenureGrp`**:
    - `Tenure Group = if [Tenure_in_Months] < 6 then "< 6 Months" else if [Tenure_in_Months] < 12 then "6-12 Months" else if [Tenure_in_Months] < 18 then "12-18 Months" else if [Tenure_in_Months] < 24 then "18-24 Months" else ">= 24 Months"`
    - `TenureGrpSorting = if [Tenure_in_Months] = "< 6 Months" then 1 else if [Tenure_in_Months] = "6-12 Months" then 2 else if [Tenure_in_Months] = "12-18 Months" then 3 else if [Tenure_in_Months] = "18-24 Months " then 4 else 5`
4. **Create New Table Reference for `prod_Services`**:
    - Unpivot services columns
    - Rename Column – `Attribute >> Services` & `Value >> Status`

## Step 3: Power BI Measure

- `Total Customers = COUNT(prod_Churn[Customer_ID])`
- `New Joiners = CALCULATE(COUNT(prod_Churn[Customer_ID]), prod_Churn[Customer_Status] = "Joined")`
- `Total Churn = SUM(prod_Churn[Churn Status])`
- `Churn Rate = [Total Churn] / [Total Customers]`

## Step 4: Power BI Visualization

1. **Summary Page**:
    - Top Card: Total Customers, New Joiners, Total Churn, Churn Rate%
    - Demographic: Gender – Churn Rate, Age Group – Total Customer & Churn Rate
    - Account Info: Payment Method – Churn Rate, Contract – Churn Rate, Tenure Group - Total Customer & Churn Rate
    - Geographic: Top 5 State – Churn Rate
    - Churn Distribution: Churn Category – Total Churn, Tooltip : Churn Reason – Total Churn
    - Service Used: Internet Type – Churn Rate, prod_Service >> Services – Status – % RT Sum of Churn Status
2. **Churn Reason Page (Tooltip)**:
    - Churn Reason – Total Churn

## Step 5: Predict Customer Churn

### What is Random Forest?
A random forest is a machine learning algorithm that consists of multiple decision trees. Each decision tree is trained on a random subset of the data and features. The final prediction is made by averaging the predictions (in regression tasks) or taking the majority vote (in classification tasks) from all the trees in the forest. This ensemble approach improves the accuracy and robustness of the model by reducing the risk of overfitting compared to using a single decision tree.

### Data Preparation for ML Model
1. **Import Views in an Excel File**:
    - Go to Data >> Get Data >> SQL Server Database
    - Enter the Server Name & Database name to connect to SQL Server
    - Import both `vw_ChurnData` & `vw_JoinData`
    - Save the file as `Prediction_Data`
2. **Create Churn Prediction Model – Random Forest**:
    - Install Anaconda: [Install Anaconda](https://docs.anaconda.com/anaconda/install/)
    - Install Libraries:
        ```sh
        pip install pandas numpy matplotlib seaborn scikit-learn joblib
        ```
    - Open Jupyter Notebook and write the following code:
        ```python
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.preprocessing import LabelEncoder
        import joblib

        # Define the path to the Excel file
        file_path = r"C:\yourpath\Prediction_Data.xlsx"
        df = pd.read_excel(file_path, sheet_name='vw_ChurnData')

        # Data Preprocessing
        columns_to_encode = ['Gender', 'Married', 'Value_Deal', 'Phone_Service', 'Multiple_Lines', 
                            'Internet_Service', 'Internet_Type', 'Online_Security', 'Online_Backup', 
                            'Device_Protection_Plan', 'Premium_Support', 'Streaming_TV', 
                            'Streaming_Movies', 'Streaming_Music', 'Unlimited_Data', 'Contract', 
                            'Paperless_Billing', 'Payment_Method']

        label_encoders = {}
        for column in columns_to_encode:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

        df['Churn'] = df['Customer_Status'].apply(lambda x: 1 if x == 'Churned' else 0)

        # Splitting the data
        X = df.drop(['Customer_ID', 'Customer_Status', 'Churn_Category', 'Churn_Reason', 'Churn'], axis=1)
        y = df['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Building the Random Forest Model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Making Predictions
        y_pred = rf_model.predict(X_test)

        # Evaluating the Model
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        # Save the model
        joblib.dump(rf_model, 'churn_model.pkl')
        ```
3. **Future Enhancements**:
    - Implement other machine learning algorithms such as Gradient Boosting or XGBoost.
    - Use more sophisticated feature engineering techniques to improve model performance.
    - Integrate real-time data processing to predict churn for new customers as soon as they join.
    - Build a web application to visualize churn predictions and key metrics.

## Conclusion
This project provides a comprehensive approach to understanding and predicting customer churn. By utilizing SQL Server for data preprocessing, Power BI for visualization, and machine learning models for prediction, businesses can gain valuable insights and take proactive steps to reduce churn and improve customer retention.
