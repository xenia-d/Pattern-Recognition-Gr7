import argparse
from GeneAnalysis import GeneAnalysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run grid search with or without cross-validation")
    parser.add_argument('--cv', type=str, default='None', help="Cross-validation method: 'No_cv' for single split, otherwise pass method name.")
    parser.add_argument('--loo_set', type=str, default='None', help="LOO set to use: 'None' for no subset, otherwise pass 1-3.")
    args = parser.parse_args()
    cv_input = args.cv
    loo_set_input = args.loo_set

    if cv_input == 'None':
        cv_input = None
    if loo_set_input == 'None':
        loo_set_input = None
    
    genes = GeneAnalysis(save_path='gridsearch_results')
    genes.run_grid_search(testing=False, cv=cv_input, loo_set=loo_set_input)
