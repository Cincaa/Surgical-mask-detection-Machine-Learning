import pandas as pd
import numpy as np
import librosa
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns


# -------------------Features-------------------

def extract_features(file_name):
    try:
        # extracting features
        # 'kaiser_fast' because is way faster and I don't see any significant improvement using 'kaiser_best'
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate)
        mfccsscaled = np.mean(mfccs.T, axis=0)

    except Exception as e:
        print("Feature extraction error: ", file_name)
        return None

    return mfccsscaled

#convert train features and labels into pd array
def extract_features_train_to_pd():
    metadata = pd.read_csv('train.txt', header=None)
    features = []

    # Iterate through each sound file and extract the features
    for index, row in metadata.iterrows():
        file_name = "train/train/" + row[0]

        class_label = row[1]
        data = extract_features(file_name)

        features.append([data, class_label])
    # Convert into a Panda dataframe
    featuresdf = pd.DataFrame(features, columns=['feature', 'label'])

    print('Finished feature extraction from ', len(featuresdf), ' files')
    return featuresdf


featuresdf = extract_features_train_to_pd()
x_train = np.array(featuresdf.feature.tolist())
train_labels = np.array(featuresdf.label.tolist())


def extract_features_validation_to_pd():
    metadata = pd.read_csv('validation.txt', header=None)
    features = []

    # Iterate through each sound file and extract the features
    for index, row in metadata.iterrows():
        file_name = "validation/validation/" + row[0]

        class_label = row[1]
        data = extract_features(file_name)

        features.append([data, class_label])
    # Convert into a Panda dataframe
    featuresdf = pd.DataFrame(features, columns=['feature', 'label'])

    print('Finished feature extraction from ', len(featuresdf), ' files')
    return featuresdf


featuresdf = extract_features_validation_to_pd()
x_validation = np.array(featuresdf.feature.tolist())
validation_labels = np.array(featuresdf.label.tolist())


def extract_features_test_to_pd():
    metadata = pd.read_csv('test.txt', header=None)
    features = []

    # Iterate through each sound file and extract the features
    for index, row in metadata.iterrows():
        file_name = "test/test/" + row[0]

        data = extract_features(file_name)

        features.append(data)

    print('Finished feature extraction from ', len(features), ' files')
    return features


features = extract_features_test_to_pd()
x_test = np.array(features)

#print predictions in .txt file
def print_results(in_file_name, out_file_name, predicted_labels_int):
    predicted_labels = predicted_labels_int.astype(str)
    file_r = open(in_file_name, 'r')
    wavs = file_r.read().splitlines()
    file_w = open(out_file_name, 'w')
    file_w.write('name,label\n')

    for i in range(len(wavs)):
        line = wavs[i] + ',' + predicted_labels[i] + '\n'
        file_w.write(line)
    file_r.close()
    file_w.close()

# plot confusion matrix
def cf_matrix(validation_labels, preds):
    cf_matrix = confusion_matrix(validation_labels, preds)
    sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Actually 0', 'Actually 1'], yticklabels=['Predicted 0', 'Predicted 1'])
    plt.show()


# accuracy, precision, recall
def report(validation_labels, preds):
    target_names = ['Without mask', 'With mask']
    print(classification_report(validation_labels, preds, target_names=target_names))


# -----GaussianNB-----
def gaussian_nb(x_train, train_labels, x_validation, validation_labels, x_test):
    clf = GaussianNB()

    clf.fit(x_train, train_labels)
    predicted_labels_int = clf.predict(x_validation)

    print('Accuracy =', clf.score(x_validation, validation_labels))

    # print_results('test_results.txt', 'test.txt',predicted_labels_int)
    cf_matrix(validation_labels, predicted_labels_int)
    report(validation_labels, predicted_labels_int)


# gaussian_nb(x_train, train_labels, x_validation, validation_labels, x_test)


# ----------------SVC----------------

def normalize(train_features, test_features, type=None):
    if type == 'min_max':
        scaler = preprocessing.MinMaxScaler()
    elif type == 'standard':
        scaler = preprocessing.StandardScaler()
    elif type == 'l1':
        scaler = preprocessing.Normalizer(norm='l1')
    elif type == 'l2':
        scaler = preprocessing.Normalizer(norm='l2')
    else:
        print("Incorrect type!")
        exit()

    scaler.fit(train_features)
    scaled_train_feats = scaler.transform(train_features)
    scaled_test_feats = scaler.transform(test_features)

    return scaled_train_feats, scaled_test_feats


scaled_tr, scaled_tst = normalize(x_train, x_validation, 'l2')

svm = svm.SVC(C=1, kernel='rbf', gamma="scale")
svm.fit(scaled_tr, train_labels)

preds = svm.predict(scaled_tst)

acc_score = accuracy_score(validation_labels, preds)
print("Accuracy: ", acc_score)
# print_results('test.txt', 'test_results_svc.txt', preds)

# confusion matrix
cf_matrix(validation_labels, preds)
report(validation_labels, preds)


#Here's my iconic fail to determine best parameters for SVC model
#It always returns the largest C and gamma and it is obviously not true
#¯\(◉‿◉)/¯
'''
from sklearn.model_selection import GridSearchCV

gsc = GridSearchCV(
        estimator=svm.SVC(kernel='rbf'),
        param_grid={
            'C': [0.1, 1, 2, 3, 4, 100, 120, 150, 1000],
            'gamma': ["scale","auto",0.0001, 0.001, 0.005, 0.1, 0.3, 0.5, 1, 10, 15],
        },

        cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

grid_result = gsc.fit(scaled_tr, train_labels)
best_params = grid_result.best_params_
print("GridSearch DONE!")
print(best_params)
'''