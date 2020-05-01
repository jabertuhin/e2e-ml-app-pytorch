import os
import sys
sys.path.append(".")
from fastapi import FastAPI
from fastapi import Path
from fastapi.responses import RedirectResponse
from http import HTTPStatus
import json
from pydantic import BaseModel
import wandb

from text_classification import config
from text_classification import data
from text_classification import predict
from text_classification import utils

app = FastAPI(
    title="text-classification",
    description="",
    version="1.0.0",
)


# Get best run
best_run = utils.get_best_run(project="GokuMohandas/e2e-ml-app-pytorch",
                              metric="test_loss", objective="minimize")

# Load best run (if needed)
best_run_dir = utils.load_run(run=best_run)

# Get run components for inference
args, model, X_tokenizer, y_tokenizer = predict.get_run_components(
    run_dir=best_run_dir)



@utils.construct_response
@app.get("/")
async def _index():
    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'data': {}
    }
    config.logger.info(json.dumps(response, indent=2))
    return response


@utils.construct_response
@app.get("/experiments")
async def _experiments():
    experiments = os.listdir(config.EXPERIMENTS_DIR)
    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'data': {"experiments": experiments}
    }
    config.logger.info(json.dumps(response, indent=2))
    return response


@utils.construct_response
@app.get("/experiment/details/{experiment_id}")
async def _experiment_details(experiment_id: str = Path(default='latest', title="ID of experiment")):
    if experiment_id == 'latest':
        experiment_id = max(os.listdir(config.EXPERIMENTS_DIR))
    experiment_dir = os.path.join(config.EXPERIMENTS_DIR, experiment_id)
    args = utils.load_json(
        filepath=os.path.join(experiment_dir, 'config.json'))
    classes = data.LabelEncoder.load(
        fp=os.path.join(experiment_dir, 'y_tokenizer.json')).classes
    performance = utils.load_json(
        filepath=os.path.join(experiment_dir, 'performance.json'))
    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'data': {"classes": classes, "args": args, "performance": performance}
    }
    config.logger.info(json.dumps(response, indent=2))
    return response


@app.get("/tensorboard")
async def _tensorboard():
    """Ensure TensorBoard is running on port 6006
    via `tensorboard --logdir tensorboard`."""
    return RedirectResponse("http://localhost:6006/")


class PredictPayload(BaseModel):
    experiment_id: str = 'latest'
    inputs: list = [{"text": ""}]


@utils.construct_response
@app.post("/predict")
async def _predict(payload: PredictPayload):
    prediction = predict.predict(inputs=payload.inputs, args=args, model=model,
                                 X_tokenizer=X_tokenizer, y_tokenizer=y_tokenizer)
    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'data': {"prediction": prediction}
    }
    config.logger.info(json.dumps(response, indent=2))
    return response
