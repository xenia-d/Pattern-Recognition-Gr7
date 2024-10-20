import argparse
from GeneAnalysis import GeneAnalysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run grid search with or without cross-validation")
    parser.add_argument('--cv', type=str, default='None', help="Cross-validation method: 'No_cv' for single split, otherwise pass method name.")
    args = parser.parse_args()
    cv_input = args.cv

    if cv_input == 'None':
        cv_input = None
    
    genes = GeneAnalysis(save_path='gridsearch_results')
    genes.run_grid_search(testing=True, cv=cv_input)
