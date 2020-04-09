
from numpy import dstack
from pandas import read_csv

from keras.utils import to_categorical


from keras.models import load_model
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

path_data = ''
name_data = 'test'
testX, testy = load_dataset_group(name_data, path_data )
print("les dimension des données de test Xtest et Ytest : ")
print(testX.shape, testy.shape)
testy = testy - 1    
testy = to_categorical(testy)
print('les dimension aprés catégorisation des Y sont : ' )
print( testX.shape, testy.shape)



 
# load model
model = load_model('modelLSTM.h5')
print("modele est chargé" )
ls, accuracy = model.evaluate(testX, testy, batch_size=64, verbose=0)
print(accuracy)
def idx_max(liste):
	mx = max(liste)
	a = 0  
	for i in range(len(liste)):
		if liste[i] == mx:
			a = i
	return  a 
 
act_predi= model.predict(testX)
name_act = ("WALKING"," WALKING_UPSTAIRS"," WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING")
for i in range(20):
    print("test #N°",i+1)
    print("activité predicté est : ",name_act[idx_max(act_predi[150*i])])
    print("activité en réalité est : ",name_act[idx_max(testy[150*i])])

print("modele est chargé" )
