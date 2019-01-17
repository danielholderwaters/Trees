import pandas as pd
# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)
#summary statistics
home_data.describe()


#average lot size (rounded to nearest integer)
avg_lot_size = round(home_data['LotArea'].mean())
print(avg_lot_size)
# As of today, how old is the newest home
import datetime
now = datetime.datetime.now()
newest_home_age = now.year - home_data['YearBuilt'].max()
print(newest_home_age)



# print the list of columns in the dataset to find the name of the prediction target
print(home_data.columns)
y = home_data.SalePrice

# Create the list of features below
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# select data corresponding to features in feature_names
X = home_data[feature_names]

# print description or statistics from X
print(X.describe())

# print the top few lines
print(X.head())


# Import the train_test_split function
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)


#Specify the model
iowa_model = DecisionTreeRegressor(random_state = 1)

# Fit iowa_model with the training data.
iowa_model.fit(train_X, train_y)

# Predict with all validation observations
val_predictions = iowa_model.predict(val_X)


# print the top few validation predictions
print(val_predictions[0:5])

# print the top few actual prices from validation data
print(val_y[0:5].values.tolist())


from sklearn.metrics import mean_absolute_error

val_mae = mean_absolute_error(val_y, val_predictions)
print(val_mae)



#lets improve the model by finding the number of leaves that would minimize MAE
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

#creating a list from 2-500 to determine which number of leaves best fits the training set
mylist = list(range(500))
candidate_max_leaf_nodes = [x+2 for x in mylist] #needed b/c there have to be at least 2 leaves
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
LeafMAE = []
for x in candidate_max_leaf_nodes:
    mae = get_mae(x, train_X, val_X, train_y, val_y)
    LeafMAE.append(mae)


#results of the optimization
results = {'Numleaves': candidate_max_leaf_nodes, 'MAE': LeafMAE}
results = pd.DataFrame(results)
results.head()
BestFit = int(results.loc[results['MAE'].idxmin()]['Numleaves'])
print(BestFit)


#lets plot these trials to visualize the minimization
import matplotlib.pyplot as plt
plt.scatter(results['Numleaves'], results['MAE'], s=1.5)
plt.xlabel("Number of Leaves")
plt.ylabel("Mean Absolute Error")
plt.show()


#Final model specification
final_model = DecisionTreeRegressor(max_leaf_nodes = BestFit, random_state=0)
final_model.fit(train_X, train_y)
preds_val = final_model.predict(val_X)
mae = mean_absolute_error(val_y, preds_val)
print(mae)


#Random Forests
from sklearn.ensemble import RandomForestRegressor
# Create X1, as to compare iteratively
features1 = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'OverallCond', 'BsmtFinSF2', 'OverallQual']
X1 = home_data[features1]
train_X1, val_X1, train_y1, val_y1 = train_test_split(X1, y, random_state=1)

# Define the model. Set random_state to 1
rf_modelA = RandomForestRegressor(random_state=1)
rf_modelB = RandomForestRegressor(random_state=1)
# fit your model
rf_modelA.fit(train_X, train_y)
rf_modelB.fit(train_X1, train_y1)

# Calculate the mean absolute error of your Random Forest model on the validation data
valA_pred = rf_modelA.predict(val_X)
valB_pred = rf_modelB.predict(val_X1)
rfA_val_mae = mean_absolute_error(valA_pred, val_y)
rfB_val_mae = mean_absolute_error(valB_pred, val_y1)
print("Validation MAE for Random Forest Model A: {}".format(rfA_val_mae))
print("Validation MAE for Random Forest Model B: {}".format(rfB_val_mae))

