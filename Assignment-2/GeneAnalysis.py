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
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, LeaveOneOut
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics import  make_scorer, accuracy_score, f1_score, confusion_matrix, roc_auc_score, silhouette_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics.cluster import normalized_mutual_info_score
import pickle

class GeneAnalysis:
    def __init__(self, data_path='Data-PR-As2/Genes', save_path='Genes_plots', random_state=12, test_size=0.2):
        self.data_path = data_path
        self.save_path = save_path
        self.random_state = 12
        self.data_normalized = None
        self.data = None
        self.labels = None
        self.import_data_and_labels()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test(test_size)
        self.y_train = self.y_train.values.ravel()
        self.y_test = self.y_test.values.ravel()
        

    def import_data_and_labels(self):
        labels = pd.read_csv(os.path.join(self.data_path,'labels.csv'))
        self.labels = labels.drop(labels='Unnamed: 0', axis=1)
        data = pd.read_csv(self.data_path+'/data.csv')
        self.data = data.drop(labels='Unnamed: 0', axis=1)
        self.data_normalized = self.get_normalized_data()
        print("Data has been loaded.")

    def get_normalized_data(self):
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
        # Convert labels to numeric
        labels_numeric, uniques = pd.factorize(self.labels.iloc[:, 0].values)  # Use first column
        
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=labels_numeric, cmap='viridis', s=50)

        # Create a colorbar with unique class labels
        cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(uniques)))
        cbar.ax.set_yticklabels(uniques)  # Set colorbar labels to the original class names

        ax.set_title('PCA Visualization of Data')
        ax.set_xlabel(f'Principal Component 1 ({explained_variance[0] * 100:.2f}% variance)')
        ax.set_ylabel(f'Principal Component 2 ({explained_variance[1] * 100:.2f}% variance)')

        if save:
            if is_normalized:
                plt.savefig(self.save_path+'/PCA_normalized.png', format='png', dpi=300)
                print(f"Plot saved as {self.save_path}/PCA_normalized.png")
            else:
                plt.savefig(self.save_path+'/PCA.png', format='png', dpi=300)
                print(f"Plot saved as {self.save_path}/PCA.png")

        plt.show(block=True)
        plt.close(fig)

    def get_mutual_info(self):
        mutual_info = mutual_info_classif(self.X_train, self.y_train.values.ravel())
        return mutual_info
    
    def get_top_k(self, k, data):
        feature_scores = pd.Series(data, index=self.X_train.columns)
        top_features = feature_scores.sort_values(ascending=False).head(k).index
        
        return top_features
    
    def plot_mutual_info(self, mut_inf, save=False, show=True):
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

    def run_kmeans(self):
        print("Running KMeans algorithm.")
        kmeans = KMeans(n_clusters = 5, random_state = 0, n_init='auto')
        kmeans.fit(self.data_normalized)

        return kmeans
    
    def evaluate_clustering(self, kmeans):
        label_encoder = LabelEncoder()
        encoded_y = label_encoder.fit_transform(self.labels['Class'])
        mu_score = normalized_mutual_info_score(encoded_y, kmeans.labels_)
        sil_score = silhouette_score(self.data_norm, kmeans.labels_, metric='euclidean')
        return sil_score, mu_score

    def plot_cluster_distribution(self, kmeans, save=True, show=True, dataset=None):
        if dataset is None:
            print("Please specify which dataset you are clustering with dataset=")
            return
        
        df = pd.DataFrame({'Label': self.labels['Class'], 'Cluster': kmeans.labels_})
        cluster_counts = df.groupby(['Cluster', 'True Label']).size().unstack(fill_value=0)

        fig, ax = plt.subplots(figsize=(8, 6))

        cluster_counts.plot(kind='bar', stacked=False, figsize=(10, 7))
        ax.set_title("True Label Distribution in Each KMeans Cluster")
        ax.set_xlabel("KMeans Cluster")
        ax.set_ylabel("Count")

        if save:
            plt.savefig(self.save_path+'/'+dataset+'_cluster_class_distribution.png', format='png', dpi=300)
            print(f"Plot saved as {self.save_path}'/'{dataset}'cluster_class_distribution.png")
        if show:
            plt.show(block=True)
        plt.close(fig)

    def plot_cluster_confusion_matrix(self, kmeans, save=True, show=True, dataset=None):
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
        ax.set_title("Contingency Table: True Labels vs KMeans Clusters")
        ax.set_xlabel("KMeans Cluster")
        ax.set_ylabel("True Label")
        if save:
            plt.savefig(self.save_path+'/'+dataset+'_cluster_confusion_matrix.png', format='png', dpi=300)
            print(f"Plot saved as {self.save_path}'/'{dataset}'_cluster_confusion_matrix.png")
        if show:
            plt.show(block=True)
        plt.close(fig)
    
    def get_pipeline_and_param_grid(self):

        # Create a pipeline with feature extraction followed by RandomForest
        pipeline = Pipeline([
            ('feature_selection', 'passthrough'),  # Step 1: Feature selection
            ('classifier', 'passthrough')  # Step 2: RandomForest
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
                'feature_selection__k': [5, 10, 20],  # Number of top features to select
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier': [rf]
            },
            # Option 1b: Use SelectKBest with Mutual Information, KNN
            {
                'feature_selection': [mutual_info],
                'feature_selection__k': [5, 10, 20],
                'classifier': [knn],
                'classifier__n_neighbors': [3, 5, 7],
                'classifier__weights': ['uniform', 'distance']
            },
            # Option 1c: Use SelectKBest with Mutual Information, SVC
            {
                'feature_selection': [mutual_info],
                'feature_selection__k': [5, 10, 20], 
                'classifier': [svc],
                'classifier__C': [0.1, 1, 10], 
                'classifier__kernel': ['linear', 'rbf']
            },
            # Option 2a: Use PCA, RF
            {
                'feature_selection': [pca],
                'feature_selection__n_components': [5, 10, 20],
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier': [rf]
            },
            # Option 2b: Use PCA, KNN
            {
                'feature_selection': [pca],
                'feature_selection__n_components': [5, 10, 20],
                'classifier': [knn],
                'classifier__n_neighbors': [3, 5, 7],
                'classifier__weights': ['uniform', 'distance'] 
            },
            # Option 2c: Use SelectKBest with Mutual Information, SVC
            {
                'feature_selection': [pca],
                'feature_selection__n_components': [5, 10, 20],
                'classifier': [svc],
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf']
            },
            # Option 3a: No feature extraction, RF
            {
                'feature_selection': ['passthrough'],
                'classifier__n_estimators': [100, 200],
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
        print("Initializing grid search pipeline and parameters.")
        # Create a pipeline with feature extraction followed by RandomForest
        pipeline = Pipeline([
            ('feature_selection', 'passthrough'),  # Step 1: Feature selection
            ('classifier', 'passthrough')  # Step 2: RandomForest
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
                'classifier': [svc],
            },
            # No feature extraction
            {
                'classifier': [svc],
            }
        ]
        
        return pipeline, param_grid


    def run_grid_search(self, testing=False):

        # Defining the cross-validation method
        cv_methods = {
            'KFold_3': KFold(n_splits=5),
            'LeaveOneOut': LeaveOneOut(),
            'No_cv': None
        }        

        # Define the scoring metrics
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1': make_scorer(f1_score, average='weighted'),
            'roc_auc': make_scorer(roc_auc_score, multi_class='ovo', average='weighted')
        }

        cv_results = {}

        if testing:
            pipeline, param_grid = self.get_pipeline_and_param_grid_TESTING_OUT()
        else:
            pipeline, param_grid = self.get_pipeline_and_param_grid()

        print("Starting grid search.")
        for cv_name, cv_strategy in cv_methods.items():
            grid_search = GridSearchCV(pipeline, param_grid, verbose=3, cv=cv_strategy, scoring=scoring, refit='f1', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            
            # Save the results for this CV strategy
            cv_results[cv_name] = {
                'grid': grid_search,
                'results': grid_search.cv_results_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_
            }
            
            self.save_to_pickle(grid_search, cv_name+'_grid_search')
        
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

    def evaluate_model_on_test_set(self, model):
        y_pred = model.predict(self.X_test)

        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(self.y_test, y_pred, multi_class='ovo', average='weighted')
        cm = confusion_matrix(self.y_test, y_pred, labels=self.labels['Class'].unique())

        return acc, f1, roc_auc, cm
    
    def plot_confusion_matrix(self, cm, model_name=None, save=False, show=False):
        if model_name is None:
            print("Please specify which model you are plotting with model_name=")
            return
        disp = ConfusionMatrixDisplay(cm, display_labels=self.labels['Class'].unique())

        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax)
        
        plt.title('Confusion Matrix')
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save:
            plt.savefig(self.save_path+'/'+model_name+'_confusion_matrix.png', format='png', dpi=300)
            print(f"Plot saved as {self.save_path}'/'{model_name}'_cluster_confusion_matrix.png")
        if show:
            plt.show(block=True)
        plt.close(fig)