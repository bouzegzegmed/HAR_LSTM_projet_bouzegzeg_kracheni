# lstm model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical

# charger un seul fichier sous forme de tableau numpy
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# charge une liste de fichiers et retourne comme un tableau numpy 3d
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	loaded = dstack(loaded)
	return loaded

# charger un groupe de jeux de données
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# charge les 9 fichiers en un seul tableau
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# charger sortie de classe 
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# chrager jeux de données 
def load_dataset(prefix=''):
	# chrager train
	trainX, trainy = load_dataset_group('train', prefix );print("les dimension des données d apprentissage Xtrain et Ytrain : ")
	print(trainX.shape, trainy.shape)
	# charger test
	testX, testy = load_dataset_group('test', prefix );print("les dimension des données de test Xtest et Ytest : ")
	print(testX.shape, testy.shape)
	# valeurs de classe à décalage nul
	trainy = trainy - 1
	testy = testy - 1
	# catégorisation des y
	trainy = to_categorical(trainy)
	testy = to_categorical(testy);print('les dimension aprés catégorisation des Y sont : ' )
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy

# ajuster et évaluer un modèle
def evaluate_model(trainX, trainy, testX, testy):
	verbose, epochs, batch_size = 0, 20, 64
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(LSTM(256, input_shape=(n_timesteps,n_features)))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	classification= model.predict(testX)
	ls, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0);model.save("modelLSTM.h5")
	return accuracy ,ls ,classification

# calcule de précision
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores) 
	print('précision moyen de model est  : %.3f%% (+/-%.3f)' % (m, s))

# fonction principale
def run_experiment(repeats=1):
	# charge des données
	trainX, trainy, testX, testy = load_dataset()
	# répétition d'expérience
	scores = list()
	for r in range(repeats):
		score ,ls ,classification = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
		print('>#précision d apprentissage N°%d: %.3f' % (r+1, score))
		scores.append(score)
	#summarize_results(scores)
	return classification , testy

# trouver indice de valeur max dans liste    
def idx_max(liste):
	mx = max(liste)
	a = 0  
	for i in range(len(liste)):
		if liste[i] == mx:
			a = i
	return  a 
 
#20 test de prédiction  
act , act_predi = run_experiment()
name_act = ("WALKING"," WALKING_UPSTAIRS"," WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING")
for i in range(20):
    print("test #N°",i+1)
    print("activité predicté est : ",name_act[idx_max(act_predi[150*i])])
    print("activité en réalité est : ",name_act[idx_max(act[150*i])])