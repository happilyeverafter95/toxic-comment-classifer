# Toxic Comment Classifier

Restructured and simplified my [solution to Kaggle's Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) for deployment.

See [Accompanying Medium Article]() for background and more detail.

## Usage Requirements

Install all requirements packages using `pip install -r requirements.txt`

This repo uses the Kaggle API to fetch the data sets. The Kaggle API requires a bit more work setting up:
1. Create a Kaggle account and [accept the competition rules](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/rules)
2. Generate an API token from `kaggle.com/USERNAME/account` (it will prompt you to download a `kaggle.json` file which contains the credentials)
3. Set Kaggle credentials as environment variables: 
```
export KAGGLE_USERNAME = [kaggle username]
export KAGGLE_KEY = [generated key]
```

## Serve the Model

Run `python profanity_detector/main.py` which fetches the data, trains the model and exports a `SavedModel` file into the `/models` path. The `SavedModel` subdirectory can be used to serve the model using [TensorFlow Serving](https://github.com/tensorflow/serving).

### Start the Server

Instructions paraphrased from the TensorFlow Serving repo.

1. Install Docker
2. Fetch the latest version of TensorFlow Serving Docker `docker pull tensorflow/serving:latest`
3. Clone the TensorFlow repo `git clone https://github.com/tensorflow/serving`
4. Specify the directory for export. In the root directory of this repo, run `ModelPath="$(pwd)/model"`

To start the server: 
```
docker run -t --rm -p 8501:8501 \
    -v "$ModelPath/models/toxic_comment_classifier" \
    -e MODEL_NAME=toxic_comment_classifier \
    tensorflow/serving
```
### Sample Curl Command

```
curl -d '{"signature_name": "predict","inputs":{"input": "raw text goes here"}}' \
  -X POST http://localhost:8501/v1/models/toxic_comment_classifier:predict
```
  
**Please note limitations of this server:**
- preprocessing steps applied to training data set is not applied to input (to be addressed later)
