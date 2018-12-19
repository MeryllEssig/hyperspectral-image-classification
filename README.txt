Ce projet se compose de 5 fichiers:
 - 4 fichiers python (train.py, utils.py, visualization.py, preprocessing.py) regroupant des fonctions que nous réutilisons dans notre notebook)
 - 1 notebook jupyter "Main.ipynb" <---- LE FICHIER A EXECUTER
 
 Les librairies requises sont :
 - Jupyter
 - tensorflow
 - Keras (tensorflow backend)
 - numpy
 - scipy (pour loader la matrice)
 - spectral (pour la visualisation)
 - matplotlib
 - scikit-learn
 - scikit-image
 
 Il est nécessaire que vos données se nomment "Indian_pines_corrected.mat", les labels d'entrainement "train_data.py", les labels de test "test_data.py", tels que nous les avons récupérés, et soient présents dans le même dossier que les fichiers sources. Si vous voulez utiliser des fichiers personnalisés, il faut modifier la fonction loadData() présente dans preprocessing.py. 
 
 Pour exécuter le programme, faites "Run" sur les cellules jupyter du fichier "Main.ipynb".