from azure.ai.ml import MLClient
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

# Authenticate and connect to the workspace
credential = InteractiveBrowserCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id="0a94de80-6d3b-49f2-b3e9-ec5818862801",
    resource_group_name="buas-y2",
    workspace_name="CV4-2025"
)

# Create and upload a Data asset
data_asset = Data(
    path="../../../data/masks",
    type=AssetTypes.URI_FOLDER,
    name="masks",
    description="Masks upload",
    version="2"
)

# Upload and register the data asset
ml_client.data.create_or_update(data_asset)
print("Dataset uploaded successfully.")
