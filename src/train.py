import mlflow #permet de suivez les experiences de machine learning 
import joblib #permet de sauvegarder et de chercher les objets python , notaments les modeles machine learning entraine

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor #ils s'agit d'un ensemble de plusieurs arbre de decision qui travaillent ensemble pour produire une predictions
from sklearn.metrics import mean_absolute_error #permet de calculer le MAE 

data  = fetch_california_housing(as_frame=True) #as_frame= True c'est a dire retourne les donnees sous forme une DataFrame 

x=data.frame.drop('MedHouseVal' , axis=1 )#les variables explicative (tous les variables except MedHouseVal) (axis=1 supprimer une column )

y = data.frame['MedHouseVal'] 

X_train , X_test , y_train , y_test = train_test_split (
    x , 
    y , 
    test_size=0.2 , 
    random_state=42
)
mlflow.set_experiment("Prediction_Immo_Californie")
with mlflow.start_run() :  #tout ce qui execute a l'interieur de ce block doit associe a ce experience cela permet a mlflow de gerer automatiquement le debut et la fin de l'execution
    model= RandomForestRegressor (
        n_estimators=100 , #signifie que notre random forest contient 100 arbres de decision 
        random_state=42 
    )
    model.fit(X_train , y_train )  

    mae = mean_absolute_error(
        y_test , 
        model.predict(X_test)
    ) #nous comparents les predictions avec les vraies valeurs 
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("MAE" , mae)  #enregistrer le metrique dans MLFLOW  
    joblib.dump(model , "model.pkl") #sauvegarder le modele avec joblib 
    print("Modèle sauvegardé avec succès !")




