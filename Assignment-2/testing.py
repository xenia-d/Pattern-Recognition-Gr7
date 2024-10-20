from GeneAnalysis import GeneAnalysis
import os

if __name__ == "__main__":
    os.chdir('Assignment-2')
    # print(os.getcwd())
    genes = GeneAnalysis()
    # genes.print_number_of_rows_and_columns()
    # genes.class_distribution_barplot(show=False, save=False)

    genes.run_grid_search(testing=True)
