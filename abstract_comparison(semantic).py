
import random
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score
from sklearn import preprocessing
import nltk
import csv

nltk.download('punkt') # for tokenization
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()

with open("social_test.txt", "r") as f:
    reader = csv.reader(f)
    testing_set  = list(reader)

testing_set = [element[0].split(" ") for element in testing_set]
# data loading and preprocessing 

# the columns of the data frame below are: 
# (1) paper unique ID (integer)
# (2) publication year (integer)
# (3) paper title (string)
# (4) authors (strings separated by ,)
# (5) name of journal (optional) (string)
# (6) abstract (string) - lowercased, free of punctuation except intra-word dashes

with open("social_train.txt", "r") as f:
    reader = csv.reader(f)
    training_set  = list(reader)

training_set = [element[0].split(" ") for element in training_set]
with open("node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info  = list(reader)

# to test code we select sample
to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set)*0.05)))
training_set = [training_set[i] for i in to_keep]
valid_ids=set()
for element in training_set:
	valid_ids.add(element[0])
	valid_ids.add(element[1])

tmp=[element for element in node_info if element[0] in valid_ids ]
node_info=tmp
del tmp



IDs = []
ID_pos={}
for element in node_info:
	ID_pos[element[0]]=len(IDs)
	IDs.append(element[0])

# we will use three basic features:

# number of overlapping words in abstract
overlap_abs = []

# temporal distance between the papers
temp_diff = []

# number of common authors
comm_auth = []



counter = 0
for i in range(len(training_set)):
    source = training_set[i][0]
    target = training_set[i][1]
    
    
    source_info = node_info[ID_pos[source]]
    target_info = node_info[ID_pos[target]]
    
    # convert to lowercase and tokenize
    source_abs= source_info[5].lower().split(" ")
	# remove stopwords
    source_abs = [token for token in source_abs if token not in stpwds]
    source_abs = [stemmer.stem(token) for token in source_abs]
    
    target_abs= target_info[2].lower().split(" ")
    target_abs= [token for token in target_abs if token not in stpwds]
    target_abs= [stemmer.stem(token) for token in target_abs]
    
    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")
    
    overlap_abs.append(len(set(source_abs).intersection(set(target_abs))))
    temp_diff.append(int(source_info[1]) - int(target_info[1]))
    comm_auth.append(len(set(source_auth).intersection(set(target_auth))))

    
    if counter % 10000 == 0:
        print(counter, "training examples processsed")
    counter += 1
		
# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
training_features = np.array([overlap_abs, temp_diff, comm_auth]).T

# scale
training_features = preprocessing.scale(training_features)

# convert labels into integers then into column array
labels = [int(element[2]) for element in training_set]
labels = list(labels)
labels_array = np.array(labels)

print("evaluating")


#evaluation
kf = KFold(len(training_set), n_folds=10)
sumf1=0
for train_index, test_index in kf:
	X_train, X_test = training_features[train_index], training_features[test_index]
	y_train, y_test = labels_array[train_index], labels_array[test_index]
	# initialize basic SVM
	classifier = svm.LinearSVC()
	# train
	classifier.fit(X_train, y_train)
	pred=classifier.predict(X_test)
	sumf1+=f1_score(pred,y_test)
	
print("\n\n")
print(sumf1/10.0)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

print ('Accuracy:', accuracy_score(y_test, pred))
print ('F1 score:', f1_score(y_test,pred,average='weighted'))
print ('Recall:', recall_score(y_test,pred,
                              average='weighted'))
print ('Precision:', precision_score(y_test, pred,
                                    average='weighted'))
print ('\n clasification report:\n', classification_report(y_test,pred))
print ('\n confussion matrix:\n',confusion_matrix(y_test,pred))

import matplotlib.pyplot as plt
def plot_classification_report(cr, title='Classification report ', with_avg_total=False, cmap=plt.cm.Blues):

    lines = cr.split('\n')

    classes = []
    plotMat = []
    for line in lines[2 : (len(lines) - 3)]:
        #print(line)
        t = line.split()
        # print(t)
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        print(v)
        plotMat.append(v)

    if with_avg_total:
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)


    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')

classificationReport=classification_report(y_test,pred)
plot_classification_report(classificationReport)

