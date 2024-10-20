from GeneAnalysis import GeneAnalysis

if __name__ == "__main__":

    genes = GeneAnalysis(save_path='gridsearch_results')
    genes.run_grid_search(testing=True)
