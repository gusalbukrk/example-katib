import time
import yaml
from kubeflow.katib import KatibClient

# Load the config
with open("experiment.yaml", "r") as f:
    experiment = yaml.safe_load(f)

# Optional: Add dynamic timestamp to metadata if needed
experiment["metadata"]["name"] = f"dt-time-series-{int(time.time())}"

client = KatibClient()
client.create_experiment(experiment, namespace="kubeflow")
print("Experiment submitted!")
