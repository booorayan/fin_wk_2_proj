
# A ML Model that Predicts Whether an Individual from East Africa Owns a Bank Account or Not

#### The project entails the creation of a simple machine learning model that predicts whether individuals situated in East Africa own a bank account or not. Owning a bank account can serve as an indicator of access to financial services and financial inclusion. Therefore, one can use the model to predict the state of financial inclusion in the regions covered in the dataset. 

#### The dataset used for this project was obtained from here http://bit.ly/FinancialDataset and it includes data from 2016 - 2018. 



#### 18/08/2019

## Requirements
The libraries required for this project included:

      pandas - for performing data analysis and cleaning.

      numpy - used for fast matrix operations.

      Matplotlib - used to create plots.

      seaborn - used for creating plots.
  
The language used was python3 and the classifier chosen for the project was the random forest classifier.  

## Description
The objective of the project is to create a machine learning model that predicts if an individual owns a bank account or not. This information (i.e. on possession of bank account) can in turn be used to predict the state of financial inclusion in a given region (East Africa). The dataset used contains information on four East African countries, Tanzania, Uganda, Kenya and Rwanda. 

Bivariate and univariate analysis were conducted on the data to help determine appropriate features. Prinicpal component analysis was used for dimensionality reduction and the Random Forests Classifier was used to create the model. 


### Experiment Design
This project followed the CRISP-DM methodology for the experiment design. The CRISP-DM methodology has the following steps:

####   1.   Problem understanding: 
Entailed gaining an understanding of the research problem and the objectives to be met for the   project. Metrics for success were also defined in this phase.
   
####   2.   Data understanding: 
Entailed the initial familiarization with and exploration of the dataset, as well as the evaluation of the quality of the dataset provided for the study. 
   
            fin.info()                           
   
####   3.   Data preparation: 
Involved data cleaning to remove missing values and ensure uniformity of data. 
   
            fin.columns = fin.columns.str.lower().str.replace(' ', '_')

            fin.rename({'the_relathip_with_head': 'rltshp_with_head', 'type_of_job':'job_type', 'level_of_educuation': 'education_level', 'has_a_bank_account':'has_bank_account', 'type_of_location':'location', 'cell_phone_access': 'cellphone_access'}, axis=1, inplace=True)

            fin.dropna(inplace=True)

            ind = fin[(fin['year'] > 2018) | (fin['education_level'] == '6')].index
            fin.drop(ind, inplace=True)
 
 The missing values were dropped because they account for a negligible percent (only 1%) of the total observations
 Rows with years greater than 2018 or an education level of 6 were dropped because they considered as anomalies

####   4.   Modelling: 
Involved the processes of selecting a model technique, selecting the features and labels for the model, generating a test design, building the model and evaluating the performance of the model. 
   
The has bank account column was selected as the target/label (i.e. y) for the model, it contains two unique values, 'yes' and 'no'
      
The features for the model (i.e. X) are all the columns excluding of the uniqueid, has_bank_account, year, and country columns.

            X = fin.drop(['uniqueid', 'year', 'has_bank_account'], 1)
            y = fin['has_bank_account']

categorical columns in the features were converted to numerical values using get_dummies()

            fin_ml = pd.get_dummies(X)

For normalization, standard scaler was used 

            from sklearn.preprocessing import StandardScaler

            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

PCA was used for dimensionality reduction.

            from sklearn.decomposition import PCA

            pca = PCA(n_components=31)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

The model chosen for this project was the random forest classifier

            # training and making predictions using RandomForestClassifier 
            from sklearn.ensemble import RandomForestClassifier

            classifier = RandomForestClassifier(max_depth=2, random_state=0)
            classifier.fit(X_train, y_train)

            # predicting the label

            pred_y = classifier.predict(X_test)
            pred_y
      
####   5.   Evaluation: 
Entailed determining the accuracy score of the model. Confusion matrix and accuracy score methods were used to determine accuracy of model
      
            from sklearn.metrics import confusion_matrix, accuracy_score

            cmatrix = confusion_matrix(y_test, pred_y)
            print(cmatrix)
            print('Accuracy is: {}' .format(accuracy_score(y_test, pred_y)))
            print('\nPercentage Accuracy with PCA is: {}%' .format(accuracy_score(y_test, pred_y) * 100))

### Conclusion

Using random Forest as the machine learning model and PCA for dimeanisonality reduction, we can correctly predict individuals with or without bank accounts with an accuracy of 86.4%.

When using LDA in place of PCA, and maintaining the random forest classifier, the accuracy of prediction increases to 88.28%

Multivariate analysis revealed that a linear regression model is not best suited for the data to be worked on.

Outliers in the dataset were not dropped because they seemed reasonable.

Most of the respondents lacked bank accounts and we would expect the model to follow the same trend.

### License

Copyright (c) 2019 **Booorayan**
  
