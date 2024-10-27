import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, LeaveOneOut, PredefinedSplit
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics import  make_scorer, accuracy_score, f1_score, confusion_matrix, roc_auc_score, silhouette_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics.cluster import normalized_mutual_info_score
from mpl_toolkits.mplot3d import Axes3D
import pickle

###### Class that handles pipeline for genes dataset 

class GeneAnalysis:
    def __init__(self, data_path='Data-PR-As2/Genes', save_path='Genes_results', random_state=12, test_size=0.2):
        self.data_path = data_path
        self.save_path = save_path
        self.random_state = random_state
        self.data_normalized = None
        self.data = None
        self.labels = None
        self.import_data_and_labels()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test(test_size)
        self.y_train = self.y_train.values.ravel()
        self.y_test = self.y_test.values.ravel()
        
    def import_data_and_labels(self):
        # Importing the data
        labels = pd.read_csv(os.path.join(self.data_path,'labels.csv'))
        self.labels = labels.drop(labels='Unnamed: 0', axis=1)
        data = pd.read_csv(self.data_path+'/data.csv')
        self.data = data.drop(labels='Unnamed: 0', axis=1)
        self.data_normalized = self.get_normalized_data()
        print("Data has been loaded.")

    def get_normalized_data(self):
        # Normalizing according to L2 norm
        normalized = normalize(self.data)
        normalized = pd.DataFrame(normalized, columns=self.data.columns)
        ("Data has been normalized.")
        return normalized

    def split_train_test(self, test_size):
        # gets train test split from NORMALIZED data
        print("Splitting train-test 80-20.")
        return train_test_split(self.data_normalized, self.labels, test_size=test_size, random_state=self.random_state)
        
    def print_number_of_rows_and_columns(self):
        rows = self.data.shape[0]
        columns = self.data.shape[1]
        
        print("Number of rows in data:", rows)
        print("Number of columns in data:", columns)

    def class_distribution_barplot(self, show=True, save=False):
        # Make a plot showing the class distributions
        class_counts = self.labels['Class'].value_counts()

        fig, ax = plt.subplots()
        sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax)
        ax.set_xlabel('Class')
        ax.set_ylabel('Frequency')
        ax.set_title('Class Distribution of Types of Tumors')

        if save:
            plt.savefig(self.save_path+'/class_distribution.png', format='png', dpi=300)
            print(f"Plot saved as {self.save_path}/class_distribution.png")
        if show:
            plt.show(block=True)
        plt.close(fig)
    
    def apply_PCA(self, use_normalized=True, plot=True, save=False):
        # Apply PCA to a dataset
        if use_normalized:
            data_numeric = self.data_normalized.apply(pd.to_numeric, errors='coerce').dropna(axis=0)
        else:
            data_numeric = self.data.apply(pd.to_numeric, errors='coerce').dropna(axis=0)

        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_numeric)
        explained_variance = pca.explained_variance_ratio_

        if plot:
            self.plot_PCA(use_normalized, data_pca, explained_variance, save)

        return data_pca, explained_variance

    def plot_PCA(self, is_normalized, data_pca, explained_variance, save=False):
        # Plot PCA to a 2D plot
        labels_numeric, uniques = pd.factorize(self.labels.iloc[:, 0].values)  # Use first column
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=labels_numeric, cmap='viridis', s=50)

        # Create a colorbar with unique class labels
        cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(uniques)))
        cbar.ax.set_yticklabels(uniques)  # Set colorbar labels to the original class names

        ax.set_title('PCA Visualization of Genes Dataset')
        ax.set_xlabel(f'PC 1 ({explained_variance[0] * 100:.2f}% variance)')
        ax.set_ylabel(f'PC 2 ({explained_variance[1] * 100:.2f}% variance)')

        if save:
            if is_normalized:
                plt.savefig(self.save_path+'/PCA_normalized.png', format='png', dpi=300)
                print(f"Plot saved as {self.save_path}/PCA_normalized.png")
            else:
                plt.savefig(self.save_path+'/PCA.png', format='png', dpi=300)
                print(f"Plot saved as {self.save_path}/PCA.png")

        plt.show(block=True)
        plt.close(fig)
    
    def plot_PCA_3D(self, is_normalized, data_pca, explained_variance, save=False):
        #Plot PCA to a 3D plot
        
        labels_numeric, uniques = pd.factorize(self.labels.iloc[:, 0].values)  # Use first column
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c=labels_numeric, cmap='viridis', s=50)
        cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(uniques)))
        cbar.ax.set_yticklabels(uniques)  # Set colorbar labels to the original class names
        
        ax.set_title('3D PCA Visualization of Genes Dataset')
        ax.set_xlabel(f'PC 1 ({explained_variance[0] * 100:.2f}% variance)')
        ax.set_ylabel(f'PC 2 ({explained_variance[1] * 100:.2f}% variance)')
        ax.set_zlabel(f'PC 3 ({explained_variance[2] * 100:.2f}% variance)')
        
        # Save the plot if required
        if save:
            if is_normalized:
                plt.savefig(self.save_path+'/PCA_3D_normalized.png', format='png', dpi=300)
                print(f"Plot saved as {self.save_path}/PCA_3D_normalized.png")
            else:
                plt.savefig(self.save_path+'/PCA_3D.png', format='png', dpi=300)
                print(f"Plot saved as {self.save_path}/PCA_3D.png")
        
        plt.show(block=True)
        plt.close(fig)

    def get_mutual_info(self):
        # Get the mutual information scores of features in the data
        mutual_info = mutual_info_classif(self.X_train, self.y_train.values.ravel())
        return mutual_info
    
    def get_top_k(self, k, data):
        # Get the top k mutual information scores of features in the data
        feature_scores = pd.Series(data, index=self.X_train.columns)
        top_features = feature_scores.sort_values(ascending=False).head(k).index
        
        return top_features
    
    def plot_mutual_info(self, mut_inf, save=False, show=True):
        # Plot barplot of mutual information scores
        mut_inf = pd.Series(mut_inf)
        fig, ax = plt.subplots(figsize=(20, 8))
        mut_inf.sort_values(ascending=False).plot.bar(ax=ax)
        ax.set_title('Sorted Bar Plot')
        ax.set_xlabel('Index')
        ax.set_ylabel('Values')

        if save:
            plt.savefig(self.save_path+'/mutual_info.png', format='png', dpi=300)
            print(f"Plot saved as {self.save_path}'/mutual_info.png'")

        if show:
            plt.show(block=True)
        plt.close(fig)

    def find_best_reduced_dataset(self, cv_method):
        # Find the best reduced dataset from the given cv_method results

        results = []
        if cv_method == 'LeaveOneOut':
            for i in [1,2,3]:
                results.append(self.load_gridsearch_results(f'Genes_results/LeaveOneOut{i}_grid_search.pkl'))
        elif cv_method == 'No_cv':
            results.append(self.load_gridsearch_results('Genes_results/No_cv_grid_search.pkl'))
        elif cv_method == 'KFold_3':
            results.append(self.load_gridsearch_results(f'Genes_results/{cv_method}_grid_search.pkl'))

        cv_results = results[0].cv_results_
        
        df = pd.DataFrame({
            'mean_fit_time': cv_results['mean_fit_time'],
            'mean_score_time': cv_results['mean_score_time'],
            'mean_test_f1': cv_results['mean_test_f1'],
            'params': cv_results['params']  
        })

        results = []

        for index, row in df.iterrows():
            params = row['params']
            
            # Check if there is a feature extraction method that is NOT 'passthrough'
            if params.get('feature_selection', 'passthrough') != 'passthrough':
                # Extract relevant data
                feature_extraction_method = params['feature_selection']
                mean_fit_time = row['mean_fit_time']
                mean_score_time = row['mean_score_time']
                f1_score = row['mean_test_f1']

                # Extract all hyperparameters related to the feature_selection step
                feature_selection_params = {key: val for key, val in params.items() if key.startswith('feature_selection__')}

                # Calculate combined score (fit + score time) and subtract the F1 score
                # Here we subtract F1 score to maximize it, as lower combined value is better
                combined_score = mean_fit_time + mean_score_time

                # Append the feature extraction method and combined score to the results list
                results.append({
                    'feature_extraction': feature_extraction_method,
                    'hyperparameters': feature_selection_params,
                    'combined_score': combined_score,
                    'f1_score': f1_score
                })
        
        # Convert the list of results to a DataFrame
        results_df = pd.DataFrame(results)

        # Find the feature extraction method with the lowest combined score
        best_method = results_df.loc[results_df['combined_score'].idxmin()]

        return best_method

    def apply_feature_reduction(self, params, plot):
        # Apply given feature reduction to original dataset
        if type(params['feature_extraction']) == type(SelectKBest()):
            mutual_info = mutual_info_classif(self.data_normalized, self.labels.values.ravel())
            feature_scores = pd.Series(mutual_info, index=self.data.columns)
            top_features = feature_scores.sort_values(ascending=False).head(params['hyperparameters']['feature_selection__k']).index
            print(top_features)
            return self.data_normalized[top_features]
        elif type(params['feature_extraction']) == type(PCA()):
            data_numeric = self.data_normalized.apply(pd.to_numeric, errors='coerce').dropna(axis=0)
            n = params['hyperparameters']['feature_selection__n_components']
            pca = PCA(n_components = n, random_state=self.random_state)
            data_pca = pca.fit_transform(data_numeric)
            explained_variance = pca.explained_variance_ratio_

            if plot:
                if n == 2:
                    self.plot_PCA(False, data_pca, explained_variance, False)
                elif n == 3:
                    self.plot_PCA_3D(False, data_pca, explained_variance, False)
                else:
                    print("Too many components to plot.")
            return data_pca

    def run_kmeans(self, data):
        print("Running KMeans algorithm.")
        kmeans = KMeans(n_clusters = 5, random_state = 0, n_init='auto')
        kmeans.fit(data)

        return kmeans
    
    def evaluate_clustering(self, kmeans):
        # Evaluate clustering on silhouette and mutual information scores
        label_encoder = LabelEncoder()
        encoded_y = label_encoder.fit_transform(self.labels['Class'])
        mu_score = normalized_mutual_info_score(encoded_y, kmeans.labels_)
        sil_score = silhouette_score(self.data_normalized, kmeans.labels_, metric='euclidean')
        print(f"Silouhette Score: {sil_score}, Mutual Information Score: {mu_score}")
        return sil_score, mu_score

    def plot_cluster_distribution(self, kmeans, save=True, show=True, dataset=None):
        # Bar plot of number of samples per class in each cluster
        if dataset is None:
            print("Please specify which dataset you are clustering with dataset=")
            return
        
        df = pd.DataFrame({'Label': self.labels['Class'], 'Cluster': kmeans.labels_})
        cluster_counts = df.groupby(['Cluster', 'Label']).size().unstack(fill_value=0)

        fig, ax = plt.subplots(figsize=(6, 4))

        cluster_counts.plot(kind='bar', stacked=False, figsize=(10, 7), ax=ax)
        ax.set_title("True Label Distribution in Each KMeans Cluster")
        ax.set_xlabel("KMeans Cluster")
        ax.set_ylabel("Count")

        if save:
            plt.savefig(self.save_path+'/'+dataset+'_cluster_class_distribution.png', format='png', dpi=300)
            print(f"Plot saved as {self.save_path}'/'{dataset}'_cluster_class_distribution.png")
        if show:
            plt.show(block=True)
        plt.close(fig)

    def plot_cluster_confusion_matrix(self, kmeans, save=True, show=True, dataset=None):
        # Make confusion matrix of clusters vs true labels 
        if dataset is None:
            print("Please specify which dataset you are clustering with dataset=")
            return
        
        label_encoder = LabelEncoder()
        encoded_y = label_encoder.fit_transform(self.labels['Class'])

        contingency_matrix = confusion_matrix(encoded_y, kmeans.labels_)

        true_label_names = label_encoder.classes_
        cluster_names = [f"Cluster {i}" for i in np.unique(kmeans.labels_)]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(contingency_matrix, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=cluster_names, yticklabels=true_label_names)
        ax.set_title(f"Contingency Table: True Labels vs KMeans Clusters for {dataset}")
        ax.set_xlabel("KMeans Cluster")
        ax.set_ylabel("True Label")
        if save:
            plt.savefig(self.save_path+'/'+dataset+'_cluster_confusion_matrix.png', format='png', dpi=300)
            print(f"Plot saved as {self.save_path}'/'{dataset}'_cluster_confusion_matrix.png")
        if show:
            plt.show(block=True)
        plt.close(fig)
    
    def get_pipeline_and_param_grid(self):
        pipeline = Pipeline([
            ('feature_selection', 'passthrough'),  # Step 1: Feature selection
            ('classifier', 'passthrough')  # Step 2: RandomForest
        ])

        # Defining Feature Extraction
        mutual_info = SelectKBest(score_func=mutual_info_classif)
        pca = PCA(random_state=self.random_state)

        # Defining classifiers
        rf = RandomForestClassifier(random_state=self.random_state)
        knn= KNeighborsClassifier()
        svc = SVC(random_state=self.random_state, probability=True)

        # Define the parameter grid
        param_grid = [
            # Option 1a: Use SelectKBest with Mutual Information, RF
            {
                'feature_selection': [mutual_info],
                'feature_selection__k': [20, 500, 1000, 5000],  # Number of top features to select
                'classifier__n_estimators': [100, 200, 500], 
                'classifier__max_depth': [None, 10, 20],
                'classifier': [rf]
            },
            # Option 1b: Use SelectKBest with Mutual Information, KNN
            {
                'feature_selection': [mutual_info],
                'feature_selection__k': [20, 500, 1000, 5000],
                'classifier': [knn],
                'classifier__n_neighbors': [3, 5, 7],
                'classifier__weights': ['uniform', 'distance']
            },
            # Option 1c: Use SelectKBest with Mutual Information, SVC
            {
                'feature_selection': [mutual_info],
                'feature_selection__k': [20, 500, 1000, 5000],
                'classifier': [svc],
                'classifier__C': [0.1, 1, 10], 
                'classifier__kernel': ['linear', 'rbf']
            },
            # Option 2a: Use PCA, RF
            {
                'feature_selection': [pca],
                'feature_selection__n_components': [3, 20, 50],
                'classifier__n_estimators': [100, 200, 500], 
                'classifier__max_depth': [None, 10, 20],
                'classifier': [rf]
            },
            # Option 2b: Use PCA, KNN
            {
                'feature_selection': [pca],
                'feature_selection__n_components': [3, 20, 50],
                'classifier': [knn],
                'classifier__n_neighbors': [3, 5, 7],
                'classifier__weights': ['uniform', 'distance'] 
            },
            # Option 2c: Use SelectKBest with Mutual Information, SVC
            {
                'feature_selection': [pca],
                'feature_selection__n_components': [3, 20, 50],
                'classifier': [svc],
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf']
            },
            # Option 3a: No feature extraction, RF
            {
                'feature_selection': ['passthrough'],
                'classifier__n_estimators': [100, 200, 500], 
                'classifier__max_depth': [None, 10, 20],
                'classifier': [rf]
            },
            # Option 3b: No feature extraction, KNN
            {
                'feature_selection': ['passthrough'],
                'classifier': [knn],
                'classifier__n_neighbors': [3, 5, 7],
                'classifier__weights': ['uniform', 'distance'] 
            },
            # Option 3c: No feature extraction, SVC
            {
                'feature_selection': ['passthrough'], 
                'classifier': [svc],
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf']
            }
        ]
        
        return pipeline, param_grid
    
    def get_loo_pipeline_and_param_grid(self, loo_set):
        # Specifically for running LeaveOneOut CV, to allow running a subset of the parameter grid
        # Create a pipeline with feature extraction followed by RandomForest
        pipeline = Pipeline([
            ('feature_selection', 'passthrough'),  # Step 1: Feature selection
            ('classifier', 'passthrough')  # Step 2: RandomForest
        ])

        # Defining classifiers
        rf = RandomForestClassifier(random_state=self.random_state)
        knn = KNeighborsClassifier()
        svc = SVC(random_state=self.random_state, probability=True)

        # Define the parameter grid
        if loo_set == 1: # Mutual info 
            # Defining Feature Extraction
            mutual_info = SelectKBest(score_func=mutual_info_classif)
            param_grid = [
                # Option 1a: Use SelectKBest with Mutual Information, RF
                {
                    'feature_selection': [mutual_info],
                    'feature_selection__k': [20, 500, 1000, 5000],  # Number of top features to select
                    'classifier__n_estimators': [100, 200, 500], 
                    'classifier__max_depth': [None, 10, 20],
                    'classifier': [rf]
                },
                # Option 1b: Use SelectKBest with Mutual Information, KNN
                {
                    'feature_selection': [mutual_info],
                    'feature_selection__k': [20, 500, 1000, 5000],
                    'classifier': [knn],
                    'classifier__n_neighbors': [3, 5, 7],
                    'classifier__weights': ['uniform', 'distance']
                },
                # Option 1c: Use SelectKBest with Mutual Information, SVC
                {
                    'feature_selection': [mutual_info],
                    'feature_selection__k': [20, 500, 1000, 5000],
                    'classifier': [svc],
                    'classifier__C': [0.1, 1, 10], 
                    'classifier__kernel': ['linear', 'rbf']
                }
            ]
        
        elif loo_set == 2: # PCA
            # Defining Feature Extraction
            pca = PCA()
            # Option 2a: Use PCA, RF
            param_grid = [{
                    'feature_selection': [pca],
                    'feature_selection__n_components': [3, 20, 50],
                    'classifier__n_estimators': [100, 200, 500], 
                    'classifier__max_depth': [None, 10, 20],
                    'classifier': [rf]
                },
                # Option 2b: Use PCA, KNN
                {
                    'feature_selection': [pca],
                    'feature_selection__n_components': [3, 20, 50],
                    'classifier': [knn],
                    'classifier__n_neighbors': [3, 5, 7],
                    'classifier__weights': ['uniform', 'distance'] 
                },
                # Option 2c: Use SelectKBest with Mutual Information, SVC
                {
                    'feature_selection': [pca],
                    'feature_selection__n_components': [3, 20, 50],
                    'classifier': [svc],
                    'classifier__C': [0.1, 1, 10],
                    'classifier__kernel': ['linear', 'rbf']
                }
            ]
        # No feature extraction
        elif loo_set == 3:
                # Option 3a: No feature extraction, RF
                param_grid = [{
                        'feature_selection': ['passthrough'],
                        'classifier__n_estimators': [100, 200, 500], 
                        'classifier__max_depth': [None, 10, 20],
                        'classifier': [rf]
                    },
                    # Option 3b: No feature extraction, KNN
                    {
                        'feature_selection': ['passthrough'],
                        'classifier': [knn],
                        'classifier__n_neighbors': [3, 5, 7],
                        'classifier__weights': ['uniform', 'distance'] 
                    },
                    # Option 3c: No feature extraction, SVC
                    {
                        'feature_selection': ['passthrough'], 
                        'classifier': [svc],
                        'classifier__C': [0.1, 1, 10],
                        'classifier__kernel': ['linear', 'rbf']
                    }
                ]
        
        return pipeline, param_grid

    def get_pipeline_and_param_grid_TESTING_OUT(self):
        # Pipeline and params for testing out the grid search before runnign the full thing

        print("Initializing grid search pipeline and parameters.")
        # Create a pipeline with feature extraction followed by RandomForest
        pipeline = Pipeline([
            ('feature_selection', 'passthrough'),  
            ('classifier', 'passthrough') 
        ])

        # Defining Feature Extraction
        mutual_info = SelectKBest(score_func=mutual_info_classif)
        pca = PCA()

        # Defining classifiers
        rf = RandomForestClassifier(random_state=self.random_state)
        knn= KNeighborsClassifier()
        svc = SVC(random_state=self.random_state)

        # Define the parameter grid
        param_grid = [
            # Option 1a: Use SelectKBest with Mutual Information, RF
            {
                'feature_selection': [mutual_info],
                'feature_selection__k': [500],
                'classifier': [rf]
            },
            # Option 1b: Use SelectKBest with Mutual Information, KNN
            {
                'feature_selection': [mutual_info],
                'feature_selection__k': [10],
                'classifier': [knn],
                'classifier__n_neighbors': [3, 5],
            },
            # Option 1c: Use SelectKBest with Mutual Information, SVC
            {
                'feature_selection': [pca],
                'feature_selection__n_components': [2],
                'classifier': [svc],
            },
            # No feature extraction
            {
                'classifier': [svc],
            }
        ]
        
        return pipeline, param_grid

    def get_cv_method_and_data(self, testing, cv):
        # Retrieving the CV Method and corresponding dataset for gridsearch

        final_X_train = self.X_train
        final_y_train = self.y_train

        if testing:
            cv_methods = {'KFold_2': KFold(n_splits=2) }
        elif cv is None:
            print("NOTE: No_cv cannot be rode in this mode. Please run it separately with cv=\'No_cv\'")
            cv_methods = {
                'KFold_3': KFold(n_splits=3),
                'LeaveOneOut': LeaveOneOut()
            }
        elif cv == 'KFold_3':
            print("CV method: KFold_3")
            cv_methods = {
                'KFold_3': KFold(n_splits=3)
            }
        elif cv == 'LeaveOneOut':
            print("CV Method: LOO")
            cv_methods = {
                'LeaveOneOut': LeaveOneOut()
            }
        elif cv == 'No_cv':
            print("CV Method: No cv")
            # Step 1: Split the dataset into 80% training and 20% validation
            X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)

            # Step 2: Combine training and validation data for GridSearchCV
            final_X_train = np.concatenate([X_train, X_val])
            final_y_train = np.concatenate([y_train, y_val])

            # Step 3: Define a validation fold
            # Mark training samples with -1 and validation samples with 0
            validation_fold = [-1] * len(X_train) + [0] * len(X_val)

            # Step 4: Define a PredefinedSplit to enforce the validation fold
            ps = PredefinedSplit(test_fold=validation_fold)

            cv_methods = {
                'No_cv': ps
            }
        return cv_methods, final_X_train, final_y_train

    def run_grid_search(self, testing=False, cv=None, loo_set=None):
        # Run the grid search

        # Defining the cross-validation method
        cv_methods, final_X_train, final_y_train = self.get_cv_method_and_data(testing, cv)

        # Define the scoring metrics
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1': make_scorer(f1_score, average='weighted')
        }

        cv_results = {}

        if testing:
            pipeline, param_grid = self.get_pipeline_and_param_grid_TESTING_OUT()
        elif cv == 'LeaveOneOut':
            pipeline, param_grid = self.get_loo_pipeline_and_param_grid(loo_set)
        else:
            pipeline, param_grid = self.get_pipeline_and_param_grid()

        print("Starting grid search.")
        for cv_name, cv_strategy in cv_methods.items():
            grid_search = GridSearchCV(pipeline, param_grid, verbose=3, cv=cv_strategy, scoring=scoring, refit='f1', n_jobs=-1)
            grid_search.fit(final_X_train, final_y_train)
            
            # Save the results for this CV strategy
            cv_results[cv_name] = {
                'grid': grid_search,
                'results': grid_search.cv_results_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_
            }
            
            self.save_to_pickle(grid_search, cv_name+'_grid_search')
        
        if cv is None:
            self.save_to_pickle(cv_results, 'all_cv_grid_search_results')

        # Print the results for each cross-validation method
        for cv_name, results in cv_results.items():
            print(f"Results for {cv_name}:")
            print(f"Best Parameters: {results['best_params']}")
            print(f"Best Score: {results['best_score']}")
            print("----------------------------")

    def save_to_pickle(self, results, filename):
        filename = os.path.join(self.save_path,filename+'.pkl')
        with open(filename, 'wb') as file:
            pickle.dump(results, file)
        print(f"Results saved to {filename}")

    def evaluate_model_on_test_set(self, params):
        # Refit model and evaluate on acc, f1, roc_auc, and confusion matrix
        pipeline, _ = self.get_pipeline_and_param_grid()
        pipeline.set_params(**params)
        model = pipeline.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)

        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovo', average='weighted', labels=np.unique(self.y_test))
        cm = confusion_matrix(self.y_test, y_pred, labels=np.unique(self.y_test))

        return acc, f1, roc_auc, cm
    
    def plot_confusion_matrix(self, cm, model_name=None, save=False, show=False):
        if save and model_name is None:
            print("Please specify which model you are plotting with model_name=")
            return
        disp = ConfusionMatrixDisplay(cm, display_labels=np.unique(self.y_test))

        fig, ax = plt.subplots(figsize=(6, 4))
        disp.plot(ax=ax)
        
        plt.title(f'Confusion Matrix for Best Model with {model_name} CV')
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save:
            plt.savefig(self.save_path+'/'+model_name+'_confusion_matrix.png', format='png', dpi=300)
            print(f"Plot saved as {self.save_path}'/'{model_name}'_cluster_confusion_matrix.png")
        if show:
            plt.show(block=True)
        plt.close(fig)
    
    def load_gridsearch_results(self, filename):
        with open(filename, 'rb') as file:
            grid_search_results = pickle.load(file)
        
        return grid_search_results

    def create_ranked_lists(self, grid_search_results, verbose=False):
        # Get a ranked list of models based on their performance in grid search

        cv_results = grid_search_results.cv_results_
        
        df = pd.DataFrame({
            'mean_fit_time': cv_results['mean_fit_time'],
            'mean_score_time': cv_results['mean_score_time'],
            'mean_test_f1': cv_results['mean_test_f1'],
            'params': cv_results['params']  
        })

        # Extract F1 scores
        f1_scores = cv_results.get('mean_test_f1', None)
        
        if f1_scores is not None:
            # Count how many have a perfect F1 score 
            perfect_f1_count = sum(1 for score in f1_scores if score == 1.0)
            if perfect_f1_count == 0:
                if verbose:
                    print("No models have F1 score of 1")
                return None, None, None, None
            if verbose:
                print(f"Number of models with a perfect F1 score: {perfect_f1_count}")


        # Filter out models that dont have perfect f1 score
        df_perfect_f1 = df[df['mean_test_f1'] == 1.0]

        # Rank models based on mean_score_time
        df_ranked_by_score_time = df_perfect_f1.sort_values(by='mean_score_time', ascending=True).reset_index(drop=True)

        # Rank models based on mean_fit_time
        df_ranked_by_fit_time = df_perfect_f1.sort_values(by='mean_fit_time', ascending=True).reset_index(drop=True)

        # Rank models based on the combined time
        df_combined = df_perfect_f1.copy()
        df_combined['combined_time'] = df_perfect_f1['mean_score_time'] + df_perfect_f1['mean_fit_time']
        df_ranked_by_combined_time = df_combined.sort_values(by='combined_time', ascending=True).reset_index(drop=True)


        return perfect_f1_count, df_ranked_by_score_time, df_ranked_by_fit_time, df_ranked_by_combined_time
    
    def plot_f1_score_distribution(self, cv_method):
        # Plot the distribution of F1 scores achieved by models in a grid search

        # Load results based on the CV method
        results = []
        if cv_method == 'LeaveOneOut':
            for i in [1, 2, 3]:
                results.append(self.load_gridsearch_results(f'Genes_results/LeaveOneOut{i}_grid_search.pkl'))
        elif cv_method == 'No_cv':
            results.append(self.load_gridsearch_results('Genes_results/No_cv_grid_search.pkl'))
        elif cv_method == 'KFold_3':
            results.append(self.load_gridsearch_results(f'Genes_results/{cv_method}_grid_search.pkl'))
        
        # Extract F1 scores from all loaded results
        f1_scores = []
        for grid_search_results in results:
            cv_results = grid_search_results.cv_results_
            f1_scores.extend(cv_results.get('mean_test_f1', []))
        
        if not f1_scores:
            print("No F1 scores found in the grid search results.")
            return

        # Create a histogram of the F1 scores
        plt.figure(figsize=(5, 3))
        plt.hist(f1_scores, bins=20, color='skyblue', edgecolor='black')

        # Add labels and title
        plt.title(f'Distribution of F1 Scores ({cv_method})')
        plt.xlabel('F1 Score')
        plt.ylabel('Frequency')

        # Show the plot
        plt.grid(True)
        plt.show()

    def find_best_model_params_for_cv_method(self, cv_method, print_all=False, verbose=False):
        # Find the best model for given cv method

        if verbose:
            print(f"######## Evaluating {cv_method} Results")
        results = []
        if cv_method == 'LeaveOneOut':
            for i in [1,2,3]:
                results.append(self.load_gridsearch_results(f'Genes_results/LeaveOneOut{i}_grid_search.pkl'))
        elif cv_method == 'No_cv':
            results.append(self.load_gridsearch_results('Genes_results/No_cv_grid_search.pkl'))
        elif cv_method == 'KFold_3':
            results.append(self.load_gridsearch_results(f'Genes_results/{cv_method}_grid_search.pkl'))
        
        best_by_combined_time={}

        for n, result in enumerate(results):  
            if verbose and cv_method == 'LeaveOneOut':   
                print(f"LOO Set {n}")   
            perfect_f1_count, ranked_by_score_time, ranked_by_fit_time, ranked_by_combined_time = self.create_ranked_lists(result, verbose)
            if ranked_by_combined_time is None:
                continue
            
            if print_all:
                # Print Ranked by Mean Score Time
                print("Ranked by Mean Score Time:")
                print(ranked_by_score_time[['mean_score_time', 'mean_test_f1', 'params']])

                # Print Ranked by Mean Fit Time
                print("\nRanked by Mean Fit Time:")
                print(ranked_by_fit_time[['mean_fit_time', 'mean_test_f1', 'params']])

                # Print Ranked by Combined Time (mean_score_time + mean_fit_time)
                print("\nRanked by Combined Time:")
                print(ranked_by_combined_time[['combined_time', 'mean_test_f1', 'params']])
            if cv_method == 'LeaveOneOut':
                best_by_combined_time[n+1] = ranked_by_combined_time.iloc[0]

        if cv_method == 'LeaveOneOut':
            min_combined_time = 1000
            best_index = -1
            best_params = None
            for n in best_by_combined_time.keys():
                if best_by_combined_time[n]['combined_time'] < min_combined_time:
                    min_combined_time = best_by_combined_time[n]['combined_time']
                    best_index = n
                    best_params = best_by_combined_time[n]
            return best_params
        else:
            return ranked_by_combined_time.iloc[0]
    
    def plot_cv_method_metrics_comparison(self, cv_methods):
        # Plot a comparison of performance over cv methods

        # Initialize lists to store metrics for each cv_method
        metrics = {'Accuracy': [], 'F1 Score': [], 'ROC AUC': []}
        cv_labels = []

        # Iterate over each cv_method to get the best model params and evaluate
        for cv_method in cv_methods:
            # Find the best model params for the cv_method
            best_params = self.find_best_model_params_for_cv_method(cv_method)
            
            # Evaluate the model on the test set
            acc, f1, roc_auc, cm = self.evaluate_model_on_test_set(best_params.params)

            # Store the results in the metrics dictionary
            metrics['Accuracy'].append(acc)
            metrics['F1 Score'].append(f1)
            metrics['ROC AUC'].append(roc_auc)
            cv_labels.append(cv_method)

        # Create the bar plot
        x = np.arange(len(cv_methods))  # The label locations
        width = 0.25  # The width of the bars

        fig, ax = plt.subplots(figsize=(6, 4))

        # Plot each metric as a separate group of bars
        ax.bar(x - width, metrics['Accuracy'], width, label='Accuracy', color='skyblue')
        ax.bar(x, metrics['F1 Score'], width, label='F1 Score', color='lightgreen')
        ax.bar(x + width, metrics['ROC AUC'], width, label='ROC AUC', color='salmon')

        # Add some text for labels, title, and custom x-axis tick labels, etc.
        ax.set_xlabel('Cross-Validation Method')
        ax.set_ylabel('Scores')
        ax.set_title('Accuracy, F1 Score, and ROC AUC for Best Models by CV Method')
        ax.set_xticks(x)
        ax.set_xticklabels(cv_labels)
        ax.legend(loc='lower right')

        # Show the plot
        plt.tight_layout()
        plt.show()
    
    def plot_best_model_across_cv_methods(self, cv_method, cv_methods):
        # Find the best model params for the input cv_method
        best_params = self.find_best_model_params_for_cv_method(cv_method)

        # Initialize lists to store metrics for each cv_method
        metrics = {'Accuracy': [], 'F1 Score': [], 'ROC AUC': []}
        cv_labels = []

        # Evaluate the model for the input cv_method
        best_acc, best_f1, best_roc_auc, _ = self.evaluate_model_on_test_set(best_params)
        
        # Store the metrics for the input cv_method
        metrics['Accuracy'].append(best_acc)
        metrics['F1 Score'].append(best_f1)
        metrics['ROC AUC'].append(best_roc_auc)
        cv_labels.append(cv_method)

        # Retrieve the same model's scores from the grid search results of other cv_methods
        for method in cv_methods:
            if method == cv_method:
                continue
            
            # Load grid search results for the other cv_method
            grid_search_results = self.load_gridsearch_results(f'Genes_results/{method}_grid_search.pkl')
            
            # Find the corresponding scores for the best model params
            cv_results = grid_search_results.cv_results_
            for i, params in enumerate(cv_results['params']):
                if params == best_params.params:  # Compare model params
                    acc = cv_results['mean_test_accuracy'][i]
                    f1 = cv_results['mean_test_f1'][i]
                    roc_auc = cv_results['mean_test_roc_auc'][i]
                    
                    # Store the metrics for this cv_method
                    metrics['Accuracy'].append(acc)
                    metrics['F1 Score'].append(f1)
                    metrics['ROC AUC'].append(roc_auc)
                    cv_labels.append(method)
                    break

        # Create the bar plot
        x = np.arange(len(cv_labels))  # The label locations
        width = 0.25  # The width of the bars

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each metric as a separate group of bars
        ax.bar(x - width, metrics['Accuracy'], width, label='Accuracy', color='skyblue')
        ax.bar(x, metrics['F1 Score'], width, label='F1 Score', color='lightgreen')
        ax.bar(x + width, metrics['ROC AUC'], width, label='ROC AUC', color='salmon')

        # Add some text for labels, title, and custom x-axis tick labels, etc.
        ax.set_xlabel('Cross-Validation Method')
        ax.set_ylabel('Scores')
        ax.set_title(f'Best Model Scores Across CV Methods (Starting with {cv_method})')
        ax.set_xticks(x)
        ax.set_xticklabels(cv_labels)
        ax.legend()

        # Show the plot
        plt.tight_layout()
        plt.show()

    def load_gridsearch_results_for_cv_method(self, cv_method):
        # Helper function to load grid search results based on the cv_method.

        results = []
        
        if cv_method == 'LeaveOneOut':
            for i in [1, 2, 3]:
                results.append(self.load_gridsearch_results(f'Genes_results/LeaveOneOut{i}_grid_search.pkl'))
        elif cv_method == 'No_cv':
            results.append(self.load_gridsearch_results('Genes_results/No_cv_grid_search.pkl'))
        elif cv_method == 'KFold_3':
            results.append(self.load_gridsearch_results(f'Genes_results/KFold_3_grid_search.pkl'))
        
        return results

    def plot_all_best_models_across_cv_methods(self, cv_methods):
        # Initialize a dictionary to store metrics for each cv_method
        results_per_method = {}
        cv_labels = []

        # Loop through each cv_method to find the best model and compare across other methods
        for cv_method in cv_methods:
            results_per_method[cv_method] = {}
            # Find the best model params for the current cv_method
            best_params = self.find_best_model_params_for_cv_method(cv_method)
            results_per_method[cv_method][cv_method] = best_params.mean_test_f1

            # Retrieve the metrics for the same model across all cv_methods
            for method in cv_methods:
                if method == cv_method:
                    continue

                # Load grid search results using the custom loading function
                grid_search_results_list = self.load_gridsearch_results_for_cv_method(method)
                
                for grid_search_results in grid_search_results_list:
                    cv_results = grid_search_results.cv_results_

                    # Find the corresponding scores for the best model params
                    for i, params in enumerate(cv_results['params']):
                        # print(params)
                        found_model = []
                        for best_param in best_params.params:
                            curr_param_found = False
                            for param in params.values():
                                if type(best_param) == (type(param)):
                                    curr_param_found = True
                                    continue
                            found_model.append(curr_param_found)

                        if False not in found_model:  # Compare model params
                            results_per_method[cv_method][method] = cv_results['mean_test_f1'][i]
                            found_model = True
                            break
                    if found_model:
                        break

                # If the model wasn't found in this cv_method, append None or default value
                if not found_model:
                    print("model not found")
                    results_per_method[cv_method][method] = None

        # Create the bar plot
        x = np.arange(len(cv_methods))  # The label locations
        width = 0.25  # The width of the bars
        fig, ax = plt.subplots(figsize=(6, 4))

        colours = ['skyblue', 'lightgreen', 'salmon']
        position = [-1, 0, 1]
        # Plot each metric as a separate group of bars
        for cv_method in cv_methods:
            for i, method in enumerate(cv_methods):
                ax.bar(x + (position[i]*width), results_per_method[cv_method][method], width, label=method, color=colours[i])

        # Add some text for labels, title, and custom x-axis tick labels, etc.
        ax.set_xlabel('Cross-Validation Method')
        ax.set_ylabel('Scores')
        ax.set_title('Comparing the Best Model per CV Method to Equivalent Model in other CV Methods')
        ax.set_xticks(x)
        ax.set_xticklabels(cv_methods)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='lower right')
        # ax.legend(loc='lower right')

        # Show the plot
        plt.tight_layout()
        plt.show()

    def find_best_reduced_dataset(self, cv_method):
        results = self.load_gridsearch_results_for_cv_method(cv_method)

        cv_results = results[0].cv_results_
        
        df = pd.DataFrame({
            'mean_fit_time': cv_results['mean_fit_time'],
            'mean_score_time': cv_results['mean_score_time'],
            'mean_test_f1': cv_results['mean_test_f1'],
            'params': cv_results['params']  
        })

        results = []

        for index, row in df.iterrows():
            params = row['params']
            
            # Check if there is a feature extraction method that is NOT 'passthrough'
            if params.get('feature_selection', 'passthrough') != 'passthrough':
                # Extract relevant data
                feature_extraction_method = params['feature_selection']
                mean_fit_time = row['mean_fit_time']
                mean_score_time = row['mean_score_time']

                # Extract all hyperparameters related to the feature_selection step
                feature_selection_params = {key: val for key, val in params.items() if key.startswith('feature_selection__')}

                # Calculate combined score (fit + score time) and subtract the F1 score
                # Here we subtract F1 score to maximize it, as lower combined value is better
                combined_score = mean_fit_time + mean_score_time

                # Append the feature extraction method and combined score to the results list
                results.append({
                    'feature_extraction': feature_extraction_method,
                    'hyperparameters': feature_selection_params,
                    'combined_score': combined_score
                })
        
        # Convert the list of results to a DataFrame
        results_df = pd.DataFrame(results)

        # Find the feature extraction method with the lowest combined score
        best_method = results_df.loc[results_df['combined_score'].idxmin()]

        return best_method
    
    def plot_feature_reduction_vs_accuracy_all_methods(self, cv_methods):
        results = {}
        
        # Iterate through each cross-validation method
        for cv_method in cv_methods:
            cv_results_list = self.load_gridsearch_results_for_cv_method(cv_method)

            # Combine results for LOO by averaging across the different runs
            pca_results = {}
            mutual_info_results = {}

            for result in cv_results_list:
                cv_results = result.cv_results_

                df = pd.DataFrame({
                    'mean_fit_time': cv_results['mean_fit_time'],
                    'mean_score_time': cv_results['mean_score_time'],
                    'mean_test_f1': cv_results['mean_test_f1'],
                    'params': cv_results['params']  
                })

                for index, row in df.iterrows():
                    params = row['params']
                    mean_test_f1 = row['mean_test_f1']

                    # Extract relevant feature extraction methods and their hyperparameters
                    feature_extraction_method = params.get('feature_selection', 'passthrough')

                    if type(feature_extraction_method) == PCA:
                        n_components = params.get('feature_selection__n_components', None)
                        if n_components is not None:
                            if n_components not in pca_results:
                                pca_results[n_components] = []
                            pca_results[n_components].append(mean_test_f1)

                    if type(feature_extraction_method) == SelectKBest:
                        k = params.get('feature_selection__k', None)
                        if k is not None:
                            if k not in mutual_info_results:
                                mutual_info_results[k] = []
                            mutual_info_results[k].append(mean_test_f1)

            # Calculate average accuracies across LOO runs (or other methods if applicable)
            pca_avg_accuracy = {k: np.mean(v) for k, v in pca_results.items()}
            mutual_info_avg_accuracy = {k: np.mean(v) for k, v in mutual_info_results.items()}

            # Store the results in a dictionary keyed by the cv_method
            results[cv_method] = {
                'pca': pca_avg_accuracy,
                'mutual_info': mutual_info_avg_accuracy
            }

        # Plotting all CV methods on the same plot
        plt.figure(figsize=(12, 5))

        # Plot PCA results
        plt.subplot(1, 2, 1)
        for cv_method, result in results.items():
            pca_keys = list(result['pca'].keys())
            pca_values = list(result['pca'].values())
            plt.plot(pca_keys, pca_values, marker='o', label=cv_method)
        
        plt.title('PCA: Number of Components vs Accuracy')
        plt.xlabel('Number of Components')
        plt.ylabel('Average F1 Score')
        plt.legend(title="CV Method")
        
        # Set y-limits and ticks based on the min and max values across all methods
        all_pca_values = [val for result in results.values() for val in result['pca'].values()]
        min_value = min(all_pca_values) * .99
        max_value = max(all_pca_values) * 1.01
        plt.ylim(bottom=min_value, top=max_value)
        plt.yticks(np.arange(min_value, max_value, step=(max_value - min_value) /5))
        plt.grid()

        # Plot Mutual Information results
        plt.subplot(1, 2, 2)
        for cv_method, result in results.items():
            mutual_info_keys = list(result['mutual_info'].keys())
            mutual_info_values = list(result['mutual_info'].values())
            plt.plot(mutual_info_keys, mutual_info_values, marker='o', label=cv_method)

        plt.title('Mutual Information: Top K Features vs Accuracy')
        plt.xlabel('Top K Features')
        plt.ylabel('Average F1 Score')
        plt.legend(title="CV Method")
        
        # Set y-limits and ticks based on the min and max values across all methods
        all_mutual_info_values = [val for result in results.values() for val in result['mutual_info'].values()]
        min_value = min(all_mutual_info_values) *.99
        max_value = max(all_mutual_info_values) * 1.01
        plt.ylim(bottom=min_value, top=max_value)
        plt.yticks(np.arange(min_value, max_value, step=(max_value - min_value) / 5))
        plt.grid()

        plt.tight_layout()
        plt.show()

    
