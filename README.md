# train-yolo-dvc

### Setup
- Clone the repo and checkout to a new branch off `main`. Do not change `main` branch
    ```bash
    #example
    git checkout -b task/train-document-layout-model main
    ```
- Install DVC extension (by iterative.ai) in VS Code
- Create python virtual environment and activate it
- Install requirements
    ```bash
    pip install -r requirements.txt
    ```
- Initialize dvc
    ```bash
    dvc init
    ```
- Verify experiment/training process by running
    ```bash
    dvc dag
    ```
- Copy yolo compatible labelled data exported from label-studio to `data/` folder. Rename the `.zip` file to `data.zip` and copy it to `data/` folder
<!-- - Update the `path` to data folder in `data/dataset.yaml` file, and class names. Note that order of class names in `data/dataset.yaml` and those in `data.zip/classes.txt` should be same -->
- Review `params.yaml` file and update parameters & hyperparameters as required
- To start training, run below command
    ```bash
    dvc exp run
    ```
- **Make sure to [commit](https://dvc.org/doc/command-reference/commit) after each experiment is complete, to track the experiment properly**
- Read dvc [documentation](https://dvc.org/doc) on different commands, options and features available
