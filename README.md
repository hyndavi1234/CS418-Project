# Electricity Usage Forecasting & Price Prediction
_Tech Stack: Python, Statsmodel, Sklearn_
<br/><br/>
Developed a machine learning model to accurately forecast energy consumption and predict its price that helps energy companies, policymakers, and consumers to make informed decisions about energy usage, pricing, and resource planning. 

## Introduction
* The objective was to develop a machine learning model that can accurately forecast electricity consumption and predict its price.
* The project will involve collecting historical data on electricity usage and pricing from data sources. This data will then be pre-processed and analyzed to identify patterns and trends in electricity consumption and pricing.
* Next, various machine learning algorithms such as time-series analysis, regression analysis, and neural networks will be applied to the data to create a predictive model.
* The model will be trained and validated using historical data, and its accuracy will be tested against new, unseen data.

## Data
The source of the data is the following link: [Dataset Source](https://data.world/houston/houston-electricity-bills).
There are 4 files, they are:
1. July 2011 to June 2012 excel file - 57,430 rows and 24 columns
2. May 2012 to April 2013 excel file - 65,806 rows and 24 columns
3. July 2012 to June 2013 excel file - 66,776 rows and 24 columns
4. July 2013 to June 2014 excel file - 67,838 rows and 24 columns
 
The following is a brief summary of the data cleaning steps we performed:
* First, we identified missing data and decided how to handle it, either by imputing the missing values or excluding the observations entirely based on the respective columns.
* Next, identified and corrected any errors and inconsistencies in the data, such as incorrect values, and formatting the date column.
* We also removed duplicate data and standardized the format of data across different tables, since we were working with multiple tables and there was overlap between the time period of the datasets which we had to account for.
* The data tables contain information regarding the building address, location, service number, billing dates, total amount due.

#### Description of each column
1. Reliant Contract No: A unique identifier for each contract.
2. Service Address: Address for the service location
3. Meter No: Meter number for the service location.
4. ESID: Electric Service Identifier for the service location.
5. Business Area: Business area code for the service location.
6. Cost Center: Cost center code for the service location.
7. Fund: Fund code for the service location.
8. Bill Type: Type of bill (e.g. "T" for "Total", "P" for "Partial", etc.).
9. Bill Date: Date the bill was generated.
10. Read Date: Date the meter was read.
11. Due Date: Due date for the bill.
12. Meter Read: Meter reading for the service location.
13. Base Cost: TBase cost for the service.
14. T&D Discretionary: Transmission and Distribution Discretionary charge for the service.
15. T&D Charges: Transmission and Distribution charge for the service.
16. Current Due: Current due amount for the service.
17. Index Charge: Index charge for the service.
18. Total Due: Total due amount for the service.
19. Franchise Fee: Franchise fee for the service.
20. Voucher Date: Date the voucher was issued for the service.
21. Billed Demand: Billed demand for the service in KVA.
22. kWh Usage: Kilowatt-hour usage for the service.
23. Nodal Cu Charge: Nodal Cu Charge for the service.
24. Adder Charge: Adder Charge for the service.

#### Statistical Data Type of Each Column
1. Reliant Contract No: integer (ratio)
2. Service Address: string (nominal)
3. Meter No: integer (nominal)
4. ESID: integer (nominal)
5. Business Area: integer (ratio))
6. Cost Center: integer (ratio)
7. Fund: integer (ratio)
8. Bill Type: string (nominal)
9. Bill Date: date (nominal)
10. Read Date: date (nominal)
11. Due Date: date (nominal)
12. Meter Read: integer (ratio)
13. Base Cost: float (nominal)
14. T&D Discretionary: float (nominal)
15. T&D Charges: float (nominal)
16. Current Due: float (nominal)
17. Index Charge: float (nominal)
18. Total Due: float (nominal)
19. Franchise Fee: float (nominal)
20. Voucher Date: date (nominal)
21. Billed Demand (KVA): integer (nominal)
22. kWh Usage: integer (nominal)
23. Nodal Cu Charge: float (nominal)
24. Adder Charge: float (nominal)

## Problem
* The key issue in generating electricity is to determine how much capacity to generate in order to meet future demand.
* Electricity usage forecasting involves predicting the demand for electricity over a specific period. This process has several uses, including energy procurement, where it helps suppliers purchase the right amount of energy to ensure a steady supply.
* The advancement of smart infrastructure and integration of distributed renewable power has raised future supply, demand, and pricing uncertainties. This unpredictability has increased interest in price prediction and energy analysis.

## Research Questions (Team Contribution)
1. Previous electricity usage data can be used for predicting the usage for future (Time-Series) - Hyndavi
2. Group areas based on their energy consumption (Clustering) - Sunil
3. Electricity usage can be predicted by using correlated features (Regression) - Sourabh
4. Classification of bill type can be done using features in the data (Classification) - Sharmisha
