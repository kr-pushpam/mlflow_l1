from pprint import pprint 
from mlflow import MlflowClient
import numpy as np 
import pandas as pd
import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


client = MlflowClient(tracking_uri='http://127.0.0.1:5000')

all_experiments= client.search_experiments()
print(all_experiments)


default_experiment=[{'name':experiment.name,'lifecycle_stage':experiment.lifecycle_stage} for experiment in all_experiments if experiment.name =='Default'][0]
print(default_experiment)

experiment_description=("it's an experiment for implementing knn")
experiment_tags={
   'project name':'KNN_Implementation',
   'area':'unsupervised learning',
   'team':'online mtech',
   'semester':'jan_2024',
   'mlflow.note.content':experiment_description
}


knn_experiment= client.create_experiment(name='knn_implementation',tags=experiment_tags)
df= pd.read_csv('dataset.csv')


#keep a track in this id
mlflow.set_tracking_uri('http://127.0.0.1:5000')
#setting the experiment on which youre going to store the model details
experiment=mlflow.set_experiment('knn_implementation')
#specify run name
run_name= 'run_knn_2'


scaler= StandardScaler
###Implementation of KNN
df= pd.get_dummies(df)
#df= scaler.fit_transform(df)

parameters={'num_clusters':4}
model = KMeans(n_clusters=parameters['num_clusters'])
clusters=model.fit_predict(df)
y_pred= clusters

#define performance metrics


inertia_score= model.inertia_
sil_score= silhouette_score(df,clusters)
metrics={
    'Inertia':inertia_score, 'Silhouette_score':sil_score
}

#logging model details to the mlflow server
artifact_path='knn_art'
#logging parameters
mlflow.log_params(parameters)
#logging performance metrics
mlflow.log_metrics(metrics)


mlflow.sklearn.log_model(sk_model=model,artifact_path=artifact_path)



with mlflow.start_run(run_name=run_name) as run:
#logging parameters
 mlflow.log_params(parameters)
#logging performance metrics
 mlflow.log_metrics(metrics)
 mlflow.sklearn.log_model(sk_model=model,artifact_path=artifact_path)


# mlflow.end_run()


