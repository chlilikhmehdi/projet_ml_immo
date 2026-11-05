import pandas as pd  #pandas permet de manipuler les donnees sous forme dataFrome 
import joblib 

def test_prediction() : #dans pytest toujours le nom de fonction principale doit contient le mot test 
    model = joblib.load('model.pkl') 

    # On crée un DataFrame avec les mêmes noms de colonnes que l'entraînement
    colonnes = [
    'MedInc',
    'HouseAge',
    'AveRooms',
    'AveBedrms',
    'Population',
    'AveOccup',
    'Latitude',
    'Longitude'
]

    sample = pd.DataFrame(
    [[8.3, 41.0, 6.98, 1.02, 322.0, 2.55, 37.88, -122.23]],
    columns=colonnes
    )
    
    prediction = model.predict(sample) 
    assert prediction[0]>0 #assert est une fonction python qui verifier ice que une condition correct ou pas 

