## Credit Risk Machine Learning Models

### Overview
The purpose of this project was to build and evaluate multiple machine learning models to assess loan risk.  The models use 85 features to predict whether a loan is a high or low risk.

#### Resources:
+ Data file:  https://kenw-data.s3.us-east-2.amazonaws.com/LoanStats_2019Q1.csv
+ Python 3.7.7
+ Scikit-learn 0.22.1
+ Imbalanced-learn 0.6.2

#### Jupyter Notebooks
+ credit_risk_resampling.ipynb:  contains all oversampling, undersampling, and combination sampling models
+ credit_risk_ensemble.ipynb:  contains all ensemble models

#### Machine Learning Models
+ Oversampling
  + Random Over Sampler
  + SMOTE
+ Undersampling
  + Cluster Centroids
+ Combination
  + SMOTEENN
+ Ensemble
  + Balanced Random Forest Classifier
  + Easy Ensemble Classifier

#### Machine Learning Model Results
<table class="table table-striped">
                        <thead class="thead-light">
                          <tr>
                            <th>Model</th>
                            <th>Balanced Accuracy Score</th>
                            <th>Precision</th>
                            <th>Recall</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            <td>Random Over Sampler</td>
                            <td>0.65</td>
                            <td>0.01</td>
                            <td>0.64</td>
                          </tr>
                          <tr>
                            <td>SMOTE</td>
                            <td>0.65</td>
                            <td>0.01</td>
                            <td>0.62</td>
                          </tr>
                          <tr>
                            <td>Cluster Centroids</td>
                            <td>0.51</td>
                            <td>0.01</td>
                            <td>0.63</td>
                          </tr>
	            <tr>
                            <td>SMOTEENN</td>
                            <td>0.66</td>
                            <td>0.01</td>
                            <td>0.74</td>
                          </tr>
            <tr>
                            <td>Balanced Random Forest Classifier</td>
                            <td>0.75</td>
                            <td>0.03</td>
                            <td>0.61</td>
                         </tr>
           <tr>
                            <td>Easy Ensemble Classifier</td>
                            <td>0.92</td>
                            <td>0.07</td>
                            <td>0.91</td>
                          </tr>
                        </tbody>
                    </table>

#### Resampling Models
The Cluster Centroids model had the lowest balanced accuracy score at 51%.  The three other resampling models all came in much closer to each other with the SMOTEENN model at 66%, and the Random Over Sampler and SMOTE models both at 65%.  Each of these models had a high number of false-positive predictions, ranging from 5.6K to 10.5K.  The high number of false-positive predictions drove the precision of each model to be 1%.  False-negative predictions came in between 23 and 33 out of the 87 total actual high-risk loans.  The high false-negative predictions lead to the recall (or sensitivity) scores ranging between 62% and 74% for the resampling models.  Based on the high number of false-positives, along with the high number of false-negatives, we can not recommend any of these models.

#### Ensemble Models
We found greater success when we tested the ensemble machine-learning models.  The Balanced Random Forest Classifier (BRFC) model returned a balanced accuracy score of 75%, a precision of 3%, and a recall of 61%, a marked improvement over the results of the resampling models.  The BRFC model still produced 34 false-negative (evidenced in the recall rate), which is concerning.  The Easy Ensemble Classifier (EEC) model was by far the best performing model.  The EEC model has a balance accuracy score of 92%, a precision of 7% (predicting 1K false-positives), and a recall of 91% (8 false-negative predictions).  We recommend the EEC model to support the decision making for loan officers and underwriters.  Additional analysis of the model may help inform the loan approval teams on the “grey-areas” to look for in an application – where the model may be giving a false-positive or false-negative reading.  

#### Model Results
##### Random Over Sampler and SMOTE
<img src="https://github.com/kenwelsh/credit-risk-machine-learning/blob/master/images/random_over_sampler_results.png" width="425"/> <img src="https://github.com/kenwelsh/credit-risk-machine-learning/blob/master/images/smote_results.png" width="425"/>

##### Cluster Centroids and SMOTEENN
<img src="https://github.com/kenwelsh/credit-risk-machine-learning/blob/master/images/cluster_centroids_results.png" width="425"/> <img src="https://github.com/kenwelsh/credit-risk-machine-learning/blob/master/images/smoteenn_results.png" width="425"/>

##### Balanced Random Forest Classifier and Easy Ensemble Classifier
<img src="https://github.com/kenwelsh/credit-risk-machine-learning/blob/master/images/bal_rndm_forest_class_results.png" width="425"/> <img src="https://github.com/kenwelsh/credit-risk-machine-learning/blob/master/images/easy_ensemble_classifier_results.png" width="425"/>
