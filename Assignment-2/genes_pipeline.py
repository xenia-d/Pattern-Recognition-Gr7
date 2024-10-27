from GeneAnalysis import GeneAnalysis

if __name__ == "__main__":
    # Initialize pipeline class
    # Automatically pre-processes data
    genes = GeneAnalysis()

    # Basic Data Analysis
    genes.print_number_of_rows_and_columns()
    genes.class_distribution_barplot(show=False, save=False)
    genes.apply_PCA()

    # Run Grid Search
    genes.run_grid_search(testing=True) # If testing = True, it runs a smaller grid search for testing (full grid search takes too long)

    # Find best model per CV method and evaluate on test set
    cv_methods = ['No_cv', 'KFold_3'] # Include all CV_methods for which you have grid search results
    for method in cv_methods:
        best = genes.find_best_model_params_for_cv_method(method, print_all=False, verbose=True) # Find best model
        print(best.params)
        acc, f1, roc, cm = genes.evaluate_model_on_test_set(best.params) # Evaluate on test set
        genes.plot_confusion_matrix(cm=cm, model_name=method, show=True) # Plot confusion matrix
        print(f"Accuracy: {acc}, F1 Score: {f1}, ROC AUC: {roc}")

    # Make comparison plots
    genes.plot_cv_method_metrics_comparison(cv_methods) # Compare best models per CV method
    genes.plot_all_best_models_across_cv_methods(cv_methods) # Compare best model per CV method to equivalent in other cv methods
    genes.plot_feature_reduction_vs_accuracy_all_methods(cv_methods) # Plot F1 score against number of features

    # Clustering on original dataset
    print("CLUSTERING ORIGINAL DATASET")
    kmeans = genes.run_kmeans(genes.data_normalized)
    result = genes.evaluate_clustering(kmeans)
    genes.plot_cluster_confusion_matrix(kmeans, save=True, show=True, dataset='Original Data')

    # Clustering on best-reduced dataset
    print("CLUSTERING BEST REDUCED DATASET")
    best = genes.find_best_reduced_dataset('No_cv') # Find the bestreduced dataset for given cv method
    print(best)
    reduced_dataset = genes.apply_feature_reduction(best, plot=True) 
    kmeans = genes.run_kmeans(reduced_dataset)
    result = genes.evaluate_clustering(kmeans)
    genes.plot_cluster_confusion_matrix(kmeans, save=True, show=True, dataset='3D PCA')