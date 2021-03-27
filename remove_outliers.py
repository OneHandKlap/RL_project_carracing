import numpy as np


def remove_outliers(x, constant):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * constant
    quartile_set = (lower_quartile - IQR, upper_quartile + IQR)
    resultList = []

    for y in a.tolist():
        if y >= quartile_set[0] and y <= quartile_set[1]:
            resultList.append(y)
    return resultList

if __name__=="__main__":
    test=[-80,-74,-55,-21,70,100,-95,-3000]
    print(test)
    print(remove_outliers(test,1.5))