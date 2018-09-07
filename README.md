

# Introduction to big data technology course - Project
Name: Or Lavi
ID: 205685498

# Notice - 
I used my local spark.

# General description
This project include code that build model that predict duration_sec
of bicycles trip.
The project also add the prediction column to the new data and 
write the result file to new csv - "201801-fordgobike-tripdata-predictions.csv"


# Workflow

1. Initialize spark context.
2. Read data from the input csv.
3. Run pre process on the data - 
    a. for each date column add those columns - add hour , year ,  month , day_in_week
    b. for each time column add those column - is_morning, is_noon, is_evening, is_night
    c. replace nulls in member_birth_year column with the average member_birth_year
    d. add age column
    e. add age_range columns - is_young, is_old ...
    f. remove rows with gender column null.
    g. make member_gender and user_type columns numbers.
    h. create distance column ( between stations)
    i. remove rows with age bigger than 110
4. build gradient_boosted_tree_regression model with 0.7 percent of the data and validate on the other 0.3 percent.
    print the RMSE and R2 of the model.
    
    the model results on test data:    
        RMSE on test data = 1528.21
        R2 on test data = 0.622555 
5. read the new data and run the model on it.
6. merge the prediction and write it to csv - "201801-fordgobike-tripdata-predictions.csv"

# running 
Run the main.py file with no parameters. ( notice - you should config to work with your own spark)
