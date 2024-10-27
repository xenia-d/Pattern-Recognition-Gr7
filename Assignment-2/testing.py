from GeneAnalysis import GeneAnalysis
import os

if __name__ == "__main__":

    genes = GeneAnalysis()
    genes.print_number_of_rows_and_columns()
    genes.class_distribution_barplot(show=False, save=False)
    # print(genes.data.rows)

    # genes.run_grid_search(testing=True)
    # cv_methods = ['LeaveOneOut', 'No_cv', 'KFold_3']
    # for method in cv_methods:
    #     best = genes.find_best_model_params_for_cv_method(method, print_all=False, verbose=True)
    #     print(best.params)
    #     acc, f1, roc, cm = genes.evaluate_model_on_test_set(best.params)
    #     genes.plot_confusion_matrix(cm=cm, model_name=method, show=True)
    #     print(f"Accuracy: {acc}, F1 Score: {f1}, ROC AUC: {roc}")

    # genes.plot_cv_method_metrics_comparison(cv_methods)
    # genes.plot_all_best_models_across_cv_methods(cv_methods)
    # genes.plot_feature_reduction_vs_accuracy_all_methods(cv_methods)

    # print("CLUSTERING ORIGINAL DATASET")
    # kmeans = genes.run_kmeans(genes.data_normalized)
    # result = genes.evaluate_clustering(kmeans)
    # genes.plot_cluster_confusion_matrix(kmeans, save=True, show=True, dataset='Original Data')

    # print("CLUSTERING BEST REDUCED DATASET")
    # best = genes.find_best_reduced_dataset('No_cv')
    # print(best)
    # reduced_dataset = genes.apply_feature_reduction(best, plot=True)
    # kmeans = genes.run_kmeans(reduced_dataset)
    # result = genes.evaluate_clustering(kmeans)
    # genes.plot_cluster_confusion_matrix(kmeans, save=True, show=True, dataset='3D PCA')

    print("CLUSTERING BEST REDUCED DATASET")
    best = genes.find_best_reduced_dataset('LeaveOneOut')
    print(best)
    reduced_dataset = genes.apply_feature_reduction(best, plot=False)
    kmeans = genes.run_kmeans(reduced_dataset)
    result = genes.evaluate_clustering(kmeans)
    genes.plot_cluster_distribution(kmeans, save=True, show=True, dataset='Top 1000 Mutual Info')
    genes.plot_cluster_confusion_matrix(kmeans, save=True, show=True, dataset='Top 1000 Mutual Info')