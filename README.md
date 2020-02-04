# Tropical-Cyclone-Status-Prediction Using Random Forests and Boosting
 I will be exploring Random Forests and Boosting on cyclone weather data from NOAA.
 
 ![alt_text](https://www.frontiersin.org/image/researchtopic/11870)

### [Data set](https://www.kaggle.com/noaa/hurricane-database)

The dataset is based on cyclone weather data from NOAA.  
You can click on the dataset link to obtain the data 

We will be predicting whether a cyclone status is a tropical depression (TD) or not.  
Status can be the following types:  
* TD – tropical depression  
* TS – tropical storm   
* HU – hurricane intensity  
* EX – Extratropical cyclone  
* SD – subtropical depression intensity  
* SS – subtropical storm intensity  
* LO – low, neither a tropical, subtropical, nor extratropical cyclone  
* WV – Tropical Wave  
* DB – Disturbance 

#### Please check the notebook for complete analysis of the dataset.

### Discussion

### Interpreting the results for the RandomForestClassifier

Random Forest Classifiers performs pretty good on both my train and validation data. I have got mean scores close to 0.90 which is good.

##### Results on Training Data:

My Model gives me an ROC AUC curve area to be 0.9626 and PR auc curve area to be 0.8083. This is evident that I have more Negative values in my data, that was the reason my model was able to predict most of the negative values which gave me higher ROC area, while having 0.8080 for PR AUC shows that my model does predict decent number of Positive values which is good as far as Training data is Considered.

Results on Testing Data:
Model gives me ROC AUC value to be 0.9011 and PR AUC to be 0.5561. These are for my validation data. I can see that My model predicts good number of negative values which gave us good ROC AUC value, but coming to the positive values model could have performed better,having 0.5561 shows that it is not doing well eith prediction of positive values.

#### Checking in terms of the context of predicting tropical depressions and the potential impact of various features.

As per the context of Predicting Tropical Depressions, My model does good job in predicting when there is no tropical Depression, but it does not do a proper job in predicting when there is a tropical depression.

And checking the Feature Importances,we can see that Latititude, Longitude and Low Wind SW, impacts our model significantly. If we have changes to these values, the prediction is impacted multi foldes and we might get varied predictions. Latitude and longitude are having similar importances, so when we consider our random forest model taking proper values of these features gives us better results.


### Selection of hyper parameters and Describing how performance changes over the hyper-parameter space.
```
RandomForestClassifier(n_estimators=100,max_leaf_nodes= 100,max_features = 0.1,bootstrap=True,
                                   n_jobs = -1)
```                                  


Coming to the Hyperparameters I have selected number of trees in the forest, n_estimators to be equal to 100, which worked better on the data, I tried different range of values, but having a value of 100 worked just fine!!

I have given the hyper parameter that grow trees with 100 leaf nodes. These are defined as a relative reduction in Impurity.

I have given a fraction value for Max_feaures which is multiplied by number of features wand that number of fetures are considered while considering best split.

Keeping Bootstrap as True gave me better results than Keeping it false, which makes whole dataset to be used while building each tree, Hyper paramters are defining my model, having better hyperparameters gave me better results.


## Describing the impact of boosting

I could see that my Ada-Boost Model performs really well on my training data, I get better results fot both ROC AU and PR AUC values. I guess the data i gave for training must have included some patterns that are not linear. Swiching to AdaBoost allows us to capture many of htese non- linear relationships which translated into better predictions of both Positve and negative examples.

But This was not the same with Validation data, Surprisingly the results were bad when compared to Random Forest Model. I think this is case of overfitting where model is performing really well on trianing data while it is not that good on validation data. Boosting did not have much impact on my validation data,we can use the Random forest model Instead. Checking the feature importances, I get to see Longitude having the highest importance, which shoes that the prediction of tropical depression is influenced by Longitude, and Latitude, while other features are not impacting the prediction much. Infact even if we remove High Wind NW , our results might not vary much as this feature is not at all impacting the prediction.

### Feature ranking:

```
Longitude         0.553081
Latitude          0.427514
Low Wind SW       0.015354
Moderate Wind NE  0.003188
Moderate Wind SE  0.000863
High Wind NW      0.000000
```

### General References
* [Guide to Jupyter](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)
* [Python Built-in Functions](https://docs.python.org/3/library/functions.html)
* [Python Data Structures](https://docs.python.org/3/tutorial/datastructures.html)
* [Numpy Reference](https://docs.scipy.org/doc/numpy/reference/index.html)
* [Numpy Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf)
* [Summary of matplotlib](https://matplotlib.org/3.1.1/api/pyplot_summary.html)
* [DataCamp: Matplotlib](https://www.datacamp.com/community/tutorials/matplotlib-tutorial-python?utm_source=adwords_ppc&utm_campaignid=1565261270&utm_adgroupid=67750485268&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=1t1&utm_creative=332661264365&utm_targetid=aud-299261629574:dsa-473406587955&utm_loc_interest_ms=&utm_loc_physical_ms=9026223&gclid=CjwKCAjw_uDsBRAMEiwAaFiHa8xhgCsO9wVcuZPGjAyVGTitb_-fxYtkBLkQ4E_GjSCZFVCqYCGkphoCjucQAvD_BwE)
* [Pandas DataFrames](https://urldefense.proofpoint.com/v2/url?u=https-3A__pandas.pydata.org_pandas-2Ddocs_stable_reference_api_pandas.DataFrame.html&d=DwMD-g&c=qKdtBuuu6dQK9MsRUVJ2DPXW6oayO8fu4TfEHS8sGNk&r=9ngmsG8rSmDSS-O0b_V0gP-nN_33Vr52qbY3KXuDY5k&m=mcOOc8D0knaNNmmnTEo_F_WmT4j6_nUSL_yoPmGlLWQ&s=h7hQjqucR7tZyfZXxnoy3iitIr32YlrqiFyPATkW3lw&e=)
* [Sci-kit Learn Trees](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree)
* [Sci-kit Learn Ensemble Models](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)
* [Sci-kit Learn Metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
* [Sci-kit Learn Model Selection](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection)
* [Sci-kit Learn Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
* [Sci-kit Learn Preprocessing](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing)
