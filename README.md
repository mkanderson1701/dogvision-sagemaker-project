# Image Classification using AWS SageMaker

This is a PyTorch image classifier built on the SageMaker platform. Using a pre-trained ResNet152 as convolutional layers, the classifier is replaced and trained using some of the optimization tools available in the SageMaker python API's.

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom. It consists of 8000+ images across 133 breeds.

## Hyperparameter Search

SageMaker's hyperparameter search is provided a range of values to try. It performs several training runs, and reports back the training loss given various combinations of parameters.

## Debugging and Performance



### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
