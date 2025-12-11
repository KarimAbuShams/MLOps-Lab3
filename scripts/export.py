import mlflow
import torch
from mlflow.tracking import MlflowClient

def export():
    mlflow.set_tracking_uri("file:./mlruns")
    client = MlflowClient()
    experiment = client.get_experiment_by_name("MLOps-Lab3-Pets")
    runs = client.search_runs(experiment.experiment_id, order_by=["metrics.train_loss ASC"])
    best_run = runs[0]
    
    model = mlflow.pytorch.load_model(f"runs:/{best_run.info.run_id}/model")
    model.to("cpu")
    model.eval()
    
    client.download_artifacts(best_run.info.run_id, "classes.json", ".")
    torch.onnx.export(model, torch.randn(1, 3, 224, 224), "model.onnx", opset_version=18, input_names=["input"], output_names=["output"])

if __name__ == "__main__":
    export()

