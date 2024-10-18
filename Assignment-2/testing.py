from GeneAnalysis import GeneAnalysis
import os

if __name__ == "__main__":
    os.chdir('Assignment-2')
    # print(os.getcwd())
    genes = GeneAnalysis()
    genes.print_number_of_rows_and_columns()
    genes.class_distribution_barplot(show=False, save=False)
    # pca, variance = genes.apply_PCA(use_normalized=True, plot=False, save=False)
    ##### conclusion: the data is already normalized ?
    mi = genes.get_mutual_info()
    top_mi = genes.get_top_k(mi, k=100)
    genes.plot_mutual_info(top_mi, show=True, save=True)


