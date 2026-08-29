import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import apyori as ap

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header= None)

# Data Preprocessing for dataset
transaction = []
for i in range(0, 7501):
    transaction.append([str(dataset.values[i, j]) for j in range(0, 20)])

# Training the apriori model on dataset
from apyori import apriori
rules = apriori(
    transactions=transaction,
    min_support=0.03,
    min_confidince=0.2,
    min_lift=3,
    min_lenght=2,
    max_length=2
)

# Visualising the results

## Displaying the first results coming directly from the output of the apriori function
results = list(rules)
results

## Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

## Displaying the results non sorted
resultsinDataFrame

## Displaying the results sorted by descending lifts
resultsinDataFrame.nlargest(n = 10, columns = 'Lift')