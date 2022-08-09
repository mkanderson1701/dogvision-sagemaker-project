# Image Classification using AWS SageMaker

This is a PyTorch image classifier built on the SageMaker platform. Using a pre-trained ResNet152 as convolutional layers, the classifier is replaced and trained using some of the optimization tools available in the SageMaker python API's.

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom. It consists of 8000+ images across 133 breeds.

## Hyperparameter Search

SageMaker's hyperparameter search is provided a range of values to try. It performs several training runs, and reports back the training loss given various combinations of parameters.

The hyperparameter search included:

* learning-rate:  (continuous range) 0.0001 - 0.02
* batch-size:  (discrete values) 32, 64, 128
* hidden-units: (discrete values) 128, 256
* dropout:  (continuous range) 0.0 - 0.5

These ranges are intentionally narrow, to reduce computing time during the search.

(https://github.com/mkanderson1701/dogvision-sagemaker-project/blob/master/hpo_tuning_2022-08-09.jpg?raw=true)

The final values ended up being LR 0.001, Batch size 64, hidden units 256, dropout 0.0. These values might vary with a more complex network or longer training.

## Debugging and Performance

Hook for the SageMaker debugging and performance profiling are added. These track a number of metrics including CPU / GPU utilization, memory, timing, and so on.

(https://github.com/mkanderson1701/dogvision-sagemaker-project/blob/master/training_job_2022-08-09.jpg?raw=true)

### Results
During my runs I found the single GPU under ml.g4dn.xlarge was underutilized. Adding num_workers=4 for the dataloader increased the network to full utilization.

Note that this metric was in early runs, and no longer an issue by the time output below was collected.

[PROFILER REPORT AVAILABLE HERE](https://github.com/mkanderson1701/dogvision-sagemaker-project/blob/master/profiler-report-sm-dbc-pytorch.html)


## Model Deployment
The model is deployed using sagemaker.pytorch.model.PyTorchModel, rather than directly from the existing estimator.

This allowed me to easily drop in a replacement (infer.py) for the deployment python script while using an existing model image. Model saves, loads, model_fn() and image preprocessing required a lot of trial and error, and retraining the image with a new train_deploy.py each time was very slow.

(https://github.com/mkanderson1701/dogvision-sagemaker-project/blob/master/inference_endpoint_2022-08-09.jpg?raw=true)
