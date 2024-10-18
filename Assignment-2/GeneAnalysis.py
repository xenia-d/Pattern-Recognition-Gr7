import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
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
        self.mutual_info = None
        

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

    def split_train_test(self):
        return train_test_split(self.data, self.labels, test_size=0.2, random_state=self.random_state)
        
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

    def apply_mutual_info(self):
        self.mutual_info = mutual_info_classif(self.X_train, self.y_train.values.ravel())
        return self.mutual_info
    
    def plot_mutual_info(self, mut_inf):
        mut_inf = pd.Series(mut_inf)
        mut_inf.sort_values(ascending=False).plot.bar(figsize=(20, 8))
        #finish saving and window etc
