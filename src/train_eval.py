# Imports for tabular model
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
import sys
import pickle as pkl
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

class train:

	def __init__(self):
		self.lrg = LogisticRegression()
		self.rf = RandomForestClassifier()
		self.boost = GradientBoostingClassifier()
		self.hgb = HistGradientBoostingClassifier()

		self.algorithms = {'lrg':self.lrg, 'rf':self.rf, 'boost':self.boost, 'hgb':self.hgb}
		self.training_df = None

	def read_and_split(self, path='../data/training.csv'):
		# Read in training data
		self.training_df = pd.read_csv(path)

		# Drop extra rows names (from incorrect overall_df writing)
		self.training_df = self.training_df.drop(self.training_df.columns[[0]], axis=1)

		# Isolate labels
		labels = self.training_df['class']
		features = self.training_df.drop('class', axis=1)

		#split data for training and testing --> 60% training, 20% validation, 20% test
		self.f_train, self.f_test, self.l_train, self.l_test = train_test_split(features, labels, test_size=.4, random_state=21)
		self.f_val, self.f_test, self.l_val, self.l_test = train_test_split(features, labels, test_size=.4, random_state=21)

	def configure_parameter(self, params, *values):
		parameters = {}
		for param, value in zip(params, values):
			parameters[param] = value
		return parameters

	# Printing function for scores and metrics
	def algo_results(self, results):
		mts = results.cv_results_['mean_test_score'] #avg score
		tol = results.cv_results_['std_test_score']  #toleranc
		optimal = results.best_params_
		print('Best parameters: {}'.format(optimal))
		for mean, std, params in zip(mts, tol, results.cv_results_['params']):
			print("%0.3f (+/-%0.3f) for %r" % (mean, std*2, params))

	# Algorithm portion 
	def train_algo(self, algo, parameters):
		cv = GridSearchCV(estimator=self.algorithms[algo], param_grid=parameters, cv=5, scoring='f1_weighted', refit=True)
		cv.fit(self.f_train, self.l_train.values.ravel())
		results=open('../data/{}output.txt'.format(algo), 'w')
		self.algo_results(cv) 
		pkl_file=open('../pkl/{}pickle.pkl'.format(algo), 'wb')
		pkl.dump(cv.best_estimator_, pkl_file)
		results.close()
		pkl_file.close()	


class evaluate:
	
	# Take in valdiation features & labels, as well as test features and labels
	def __init__(self, f_val, l_val, f_test, l_test):
		self.names = ['rf']
		self.algo_pkl = {}
		self.f_val = f_val
		self.l_val = l_val
		self.f_test = f_test
		self.l_test = l_test

	# Grab the pickled models from disk	
	def get_pkls(self):
		for name in self.names: #get pkl objects
			pklfile = open('../pkl/{}pickle.pkl'.format(name), 'rb') #opens pickle and stores it in an array  
			self.algo_pkl[name] = pkl.load(pklfile)
			pklfile.close()

	# Get precision and recall values over the epochs for the validation set
	def pr_analysis(self):
		prec = {}
		reca = {}
		for name in self.names:
			prob = self.algo_pkl[name].predict_proba(self.f_val)[:,1]
			precision, recall, _ = precision_recall_curve(self.l_val, prob)
			prec[name] = precision
			reca[name] = recall
		return prec, reca 

	# Compute validation or test accuracy and f_score for each model
	def test(self, typ):
		features = None
		labels = None

		if(typ == 'val'):
			features = self.f_val
			labels = self.l_val
		elif(typ == 'test'):
			features = self.f_test
			labels = self.l_test
		else:
			print("Please choose either 'val' for validation or 'test' for test")

		# Calculate acc and f-score
		for name in self.names:
			print("Score for {name}") #assigning test score
			prediction = self.algo_pkl[name].predict(features)
			print(accuracy_score(labels, prediction))
			print(f1_score(labels, prediction, average='weighted'))


if __name__ == '__main__':

	# Example usage : training a random forest and evaluating it on the validation set

	tr = train()
	tr.read_and_split('../data/training.csv')
	params = tr.configure_parameter(
		['n_estimators', 'max_depth'], 
		[6, 8], [16, 22, 36])
	tr.train_algo('rf', params)		

	ev = evaluate(tr.f_val, tr.l_val, tr.f_test, tr.l_test)
	ev.get_pkls()
	ev.test('val')