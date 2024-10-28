
# Task 1

## Genes Dataset Pipeline
1. Install requirements
2. Run genes_pipeline.py to run an example pipeline. Note that the grid search in this file is set to a smaller grid search, for ``testing'', as the full grid search consists of over 150 models and, depending on the cross-validation method, may use leave-one-out cross-validation. Please read the comments in the genes_pipeline file for further clarification.
3. Results of the grid search can be found in Genes_results folder, but note that the Leave-One-Out cross-validation results are not available as the file was too large to push to github.
4. Plots are saved in Genes_results
5. To run full grid search on habrok, you can use the files in the habrok_jobs folder, which make use of the run_gridsearch.py file.

## BigCats Dataset Pipeline
This task includes the notebooks: image_figure.ipynb, image_classification.ipynb, and image_clustering.ipynb
1. Install required packages
2. Create local directory under Assignment-2/data for the dataset, since the image dataset is too large to upload to github 
3. Inside Assignment-2/data decompress the dataset so that the folder structure is resembles what is shown below: 

data
|-- BigCats
    |-- Cheetah
    |-- Jaguar
    |-- Leopard
    |-- Lion
    |-- Tiger

4. Run the notebooks in any order will be fine. 
5. image_classification.ipynb and image_clustering.ipynb each contains a grid search cell the longest grid search ran from the coder's computer is 25 mins. 
6. figures can be seen from the 'figures' folder. 
 
# Task 2

The code is all found in the file 'semi-supervised.py'. You can either run this on your local device, or to run on Habrok see the 'habrok_run_task_2.sh' file under the folder 'habrok_jobs'
