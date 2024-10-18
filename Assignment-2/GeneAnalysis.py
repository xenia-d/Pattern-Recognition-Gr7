import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

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
        

    def import_data_and_labels(self):
        labels = pd.read_csv(os.path.join(self.data_path,'labels.csv'))
        self.labels = labels.drop(labels='Unnamed: 0', axis=1)
        data = pd.read_csv(self.data_path+'/data.csv')
        self.data = data.drop(labels='Unnamed: 0', axis=1)
        self.data_normalized = self.get_normalized_data()

    def get_normalized_data(self):
        normalized = preprocessing.normalize(self.data)
        normalized = pd.DataFrame(normalized, columns=self.data.columns)
        return normalized

    def split_train_test(self, test_size):
        return train_test_split(self.data, self.labels, test_size=test_size, random_state=self.random_state)
        
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
        #  Drop non-numeric columns
        if use_normalized:
            data_numeric = self.data_normalized.apply(pd.to_numeric, errors='coerce').dropna(axis=0)
        else:
            data_numeric = self.data.apply(pd.to_numeric, errors='coerce').dropna(axis=0)

        # Applying PCA to project data onto 2D
        pca = PCA(n_components=2)

        data_pca = pca.fit_transform(data_numeric)
        explained_variance = pca.explained_variance_ratio_

        if plot:
            self.plot_PCA(use_normalized, data_pca, explained_variance, save)

        return data_pca, explained_variance


    def plot_PCA(self, is_normalized, data_pca, explained_variance, save=False):
        # Convert labels to numeric
        labels_numeric, uniques = pd.factorize(self.labels.iloc[:, 0].values)  # Use first column
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create the scatter plot with the PCA data and class labels
        scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=labels_numeric, cmap='viridis', s=50)

        # Create a colorbar with unique class labels
        cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(uniques)))
        cbar.ax.set_yticklabels(uniques)  # Set colorbar labels to the original class names

        # Set axis labels with explained variance percentages
        ax.set_title('PCA Visualization of Data')
        ax.set_xlabel(f'Principal Component 1 ({explained_variance[0] * 100:.2f}% variance)')
        ax.set_ylabel(f'Principal Component 2 ({explained_variance[1] * 100:.2f}% variance)')

        # Save the plot if a save_path is provided
        if save:
            if is_normalized:
                plt.savefig(self.save_path+'/PCA_normalized.png', format='png', dpi=300)
                print(f"Plot saved as {self.save_path}/PCA_normalized.png")
            else:
                plt.savefig(self.save_path+'/PCA.png', format='png', dpi=300)
                print(f"Plot saved as {self.save_path}/PCA.png")

        # Show the plot in a window
        plt.show(block=True)

        # Close the figure after it's closed by the user
        plt.close(fig)

    def get_mutual_info(self):
        mutual_info = mutual_info_classif(self.X_train, self.y_train.values.ravel())
        return mutual_info
    
    def get_top_k(self, k, data):
        # Create a DataFrame associating features with their scores
        feature_scores = pd.Series(data, index=self.X_train.columns)
        
        # Sort the features by their scores in descending order
        top_features = feature_scores.sort_values(ascending=False).head(k).index
        
        return top_features
    
    def plot_mutual_info(self, mut_inf, save=False, show=True):
        mut_inf = pd.Series(mut_inf)

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(20, 8))

        # Sort the DataFrame/Series and plot it as a bar chart
        mut_inf.sort_values(ascending=False).plot.bar(ax=ax)

        # Set labels and title if desired (optional)
        ax.set_title('Sorted Bar Plot')
        ax.set_xlabel('Index')
        ax.set_ylabel('Values')

        # Save the plot if a save_path is provided
        if save:
            plt.savefig(self.save_path+'/mutual_info.png', format='png', dpi=300)
            print(f"Plot saved as {self.save_path}'/mutual_info.png'")

        # Show the plot in a window
        if show:
            plt.show(block=True)

        # Close the figure after it's closed by the user
        plt.close(fig)

    def run_grid_search(self):
        # Feature extraction step (SelectKBest using mutual information)
        mutual_info = SelectKBest(score_func=mutual_info_classif)
        pca = PCA()

        # Random Forest classifier
        rf = RandomForestClassifier(random_state=self.random_state)

        # Create a pipeline with feature extraction followed by RandomForest
        pipeline = Pipeline([
            ('feature_selection', 'passthrough'),  # Step 1: Feature selection
            ('classifier', 'passthrough')  # Step 2: RandomForest
        ])

        # Define the parameter grid
        param_grid = [
            # Option 1: Use SelectKBest with Mutual Information, RF
            {
                'feature_selection': [mutual_info],
                'feature_selection__k': [5, 10, 20],  # Number of top features to select
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier': [rf],
            },
            # Option 1: Use SelectKBest with Mutual Information, KNN
            {
                'feature_selection': [mutual_info],
                'feature_selection__k': [5, 10, 20],  # Number of top features to select
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier': [knn],
            },
            # Option 2: Use PCA for feature extraction
            {
                'feature_selection': [pca],
                'feature_selection__n_components': [5, 10, 20],  # Number of PCA components
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20],
            },
            # Option 3: No feature extraction (use all features)
            {
                'feature_selection': ['passthrough'],  # Use all features without any extraction
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20],
            }
        ]

        # Set up the grid search
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

        # Fit the grid search on your training data
        grid_search.fit(self.X_train, self.y_train)

        # Get the best parameters and model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        print("Best Parameters:", best_params)
        print("Best Model:", best_model)