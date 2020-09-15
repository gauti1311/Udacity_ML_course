#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression().fit(ages,net_worths)
    predict = list(reg.predict(ages))
    error = [abs(predict[i]-net_worths[i]) for i in range(len(net_worths))]
    
    for i in range(len(ages)):
        cleaned_data.append((ages[i],net_worths[i],error[i]))
    
    cleaned_data.sort(key=lambda cleaned_data:cleaned_data[2], reverse=True)
    
    del cleaned_data[0:9]

    return cleaned_data

