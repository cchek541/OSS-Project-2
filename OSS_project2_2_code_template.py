import pandas as pd
from sklearn.impute import SimpleImputer
#from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv(r"C:\Users\cchek\OneDrive\Desktop\subject\2-2\Intro to open source sw\oss project 2\2019_kbo_for_kaggle_v2.csv")

def sort_dataset(dataset_df):
	sortedDf = df.sort_values(by = 'year')
	return sortedDf

def split_dataset(dataset_df):	
	df['salary'] = df['salary'] * 0.001
	trainDf = df.iloc[:1718]
	testDf = df.iloc[1718:]

	X_train = trainDf.drop('salary', axis = 1)
	Y_train = trainDf['salary']

	X_test = testDf.drop('salary', axis = 1)
	Y_test = testDf['salary']

	return X_train, X_test, Y_train, Y_test

def extract_numerical_cols(dataset_df):
	numCols = dataset_df.select_dtypes(include = 'number')

	mean = SimpleImputer(strategy = 'mean')
	numCols_imputed = pd.DataFrame(mean.fit_transform(numCols), columns = numCols.columns)
	return numCols_imputed

def train_predict_decision_tree(X_train, Y_train, X_test):
	dtm = DecisionTreeRegressor()
	dtm.fit(X_train, Y_train)
	dtm_predictions = dtm.predict(X_test)
	return dtm_predictions

def train_predict_random_forest(X_train, Y_train, X_test):
	rfm = RandomForestRegressor()
	rfm.fit(X_train, Y_train)
	rfm_predictions = rfm.predict(X_test)
	return rfm_predictions

def train_predict_svm(X_train, Y_train, X_test):
	svm = SVR()
	svm.fit(X_train, Y_train)
	svm_predictions = svm.predict(X_test)
	return svm_predictions

def calculate_RMSE(labels, predictions):
	rmse = mean_squared_error(labels, predictions, squared = False)
	return rmse

if __name__=='__main__':
	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
	
	sorted_df = sort_dataset(data_df)	
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
	
	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)
	
	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))	
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))	
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))