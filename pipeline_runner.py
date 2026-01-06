import os
from datetime import datetime
from kfp import dsl, compiler
from kfp.dsl import Output, Input, Dataset, Model, Metrics, component
from google.cloud import aiplatform, bigquery, storage
import json
import sys
import traceback
import pandas as pd
import kagglehub
from pathlib import Path
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# ============================================================================
# COMPONENTS DEFINITION
# ============================================================================

@component(
    base_image="gcr.io/deeplearning-platform-release/base-cpu:latest",
    packages_to_install=[
        "google-cloud-bigquery==3.11.4",
        "google-cloud-storage==2.10.0",
        "pandas==2.0.3",
        "scikit-learn==1.3.0",
        "pyarrow==12.0.1",
        "db-dtypes==1.1.1",
        "kagglehub"
    ]
)
def setup_and_prepare_data(
    project_id: str,
    location: str,
    bucket_name: str,
    dataset_name: str,
    table_name: str,
    git_commit_sha: str, # New parameter for versioning
    dataset_out: Output[Dataset]
):
    """Setup GCP resources and load California housing dataset"""
    import os
    import sys
    import traceback
    from google.cloud import bigquery, storage
    import pandas as pd
    import json
    import kagglehub
    from pathlib import Path
    import glob

    os.environ['GOOGLE_CLOUD_PROJECT'] = project_id

    print("="*70)
    print("COMPONENT 1: SETUP & DATA PREPARATION")
    print("="*70)
    try:
        def getRawData():
            # Download latest version
            path = kagglehub.dataset_download("camnugent/california-housing-prices")
            # Folder containing CSV files (set via env var or change here)
            CSV_FOLDER = path
            data_path = Path(CSV_FOLDER)
            csv_files = list(data_path.glob('*.csv')) if data_path.exists() else []
            if csv_files:
                print(f"   Found {len(csv_files)} CSV file(s) in '{data_path}'. Loading...")
                dfs = []
                for p in csv_files:
                    try:
                        print(f"    - Reading {p}")
                        dfs.append(pd.read_csv(p))
                    except Exception as e:
                        print(f"    ! Failed to read {p}: {e}")
                if dfs:
                    # Concatenate, allowing for differing columns
                    df = pd.concat(dfs, ignore_index=True, sort=False)
                    print(f"   Loaded combined dataframe with shape: {df.shape}")
                else:
                    print("   No valid CSVs loaded; falling back to sklearn fetch.")
            else:
                print(f"   No CSV files found in '{data_path}'.")
            return df

        print("\n[1/6] Initializing GCP clients...")
        bq_client = bigquery.Client(project=project_id)
        storage_client = storage.Client(project=project_id)
        print("   Clients initialized")

        print("\n[2/6] Setting up BigQuery dataset...")
        dataset_id = f"{project_id}.{dataset_name}"
        try:
            dataset = bq_client.get_dataset(dataset_id)
            print(f"   Using existing dataset: {dataset_id}")
        except:
            dataset = bigquery.Dataset(dataset_id)
            dataset.location = location
            dataset = bq_client.create_dataset(dataset, exists_ok=True)
            print(f"   Created dataset: {dataset_id}")

        print("\n[3/6] Setting up Cloud Storage bucket...")
        try:
            bucket = storage_client.get_bucket(bucket_name)
            print(f"   Using existing bucket: gs://{bucket_name}")
        except:
            bucket = storage_client.create_bucket(bucket_name, location=location)
            print(f"   Created bucket: gs://{bucket_name}")

        print("\n[4/6] Loading California housing dataset...")
        df = getRawData()
        print(f"   Loaded {len(df):,} rows with {len(df.columns)} columns")

        print("\n[5/6] Uploading data to BigQuery...")
        table_id = f"{project_id}.{dataset_name}.{table_name}"
        bq_client.delete_table(table_id, not_found_ok=True)

        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        job = bq_client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()

        print(f"   Uploaded to: {table_id}")

        # Save processed dataframe to GCS for versioning
        versioned_data_path = f"data/{git_commit_sha}/processed_data.csv"
        bucket.blob(versioned_data_path).upload_from_string(df.to_csv(index=False), 'text/csv')
        versioned_data_uri = f"gs://{bucket_name}/{versioned_data_path}"
        print(f"   Versioned data uploaded to GCS: {versioned_data_uri}")

        metadata = {
            "table_id": table_id,
            "num_rows": int(len(df)),
            "num_features": int(len(df.columns) - 1),
            "feature_columns": [str(col) for col in df.columns[:-1]],
            "target_column": str(df.columns[-1]),
            "gcs_data_uri": versioned_data_uri # Store GCS URI in metadata
        }

        os.makedirs(os.path.dirname(dataset_out.path), exist_ok=True)
        with open(dataset_out.path, 'w') as f:
            json.dump(metadata, f, indent=2)

        dataset_out.metadata.update(metadata)
        dataset_out.uri = versioned_data_uri # Set output artifact URI to the GCS path

        print("\n COMPONENT 1 COMPLETE")
        print("="*70)

    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

@component(
    base_image="gcr.io/deeplearning-platform-release/base-cpu:latest",
    packages_to_install=[
        "google-cloud-bigquery==3.11.4",
        "google-cloud-aiplatform>=1.38.0",
        "scikit-learn==1.3.0",
        "pandas==2.0.3"
    ]
)
def train_and_register_model(
    project_id: str,
    location: str,
    dataset_name: str,
    table_name: str,
    bucket_name: str,
    model_display_name: str,
    model_description: str,
    git_commit_sha: str, # Existing parameter
    git_author: str,
    git_commit_message: str,
    dataset_in: Input[Dataset],
    model_out: Output[Model]
):
    """Trains a model and registers it to Vertex AI Model Registry."""
    import os
    import pandas as pd
    from google.cloud import bigquery, aiplatform, storage
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import joblib
    import json

    print("="*70)
    print("COMPONENT 2: MODEL TRAINING AND REGISTRATION")
    print("="*70)

    aiplatform.init(project=project_id, location=location)

    print(f"\n[0/6] Starting Vertex AI Experiment Run for '{model_display_name}'...")
    with aiplatform.start_run(experiment=model_display_name):
        aiplatform.log_params({
            "git_commit_sha": git_commit_sha,
            "git_author": git_author,
            "git_commit_message": git_commit_message
        })
        print(f"   Logged Git metadata: SHA={git_commit_sha}, Author={git_author}, Message='{git_commit_message}'")

        print(f"\n[1/6] Reading data from dataset_in: {dataset_in.uri}")
        # Now dataset_in.uri points to the versioned GCS path
        bq_client = bigquery.Client(project=project_id)

        # Read metadata to get the original BigQuery table_id (as fallback or for full dataset)
        with open(dataset_in.path, 'r') as f:
            metadata = json.load(f)
        table_id = metadata['table_id']

        # For training, let's continue to use the full BigQuery table as the canonical source
        # If the intention was to use the *versioned GCS data*, that would require reading from dataset_in.uri
        # df = pd.read_csv(dataset_in.uri) # If we wanted to use the versioned GCS CSV for training
        print(f"\n[2/6] Fetching data from BigQuery table: {table_id} (original source)")
        df = bq_client.query(f"SELECT * FROM `{table_id}`").to_dataframe()
        print(f"   Fetched {len(df):,} rows.")

        print("\n[3/6] Preprocessing data...")
        df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)
        df = df.dropna()

        X = df.drop('median_house_value', axis=1)
        y = df['median_house_value']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"   Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

        print("\n[4/6] Training RandomForestRegressor model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print(f"   Model trained. RMSE: {rmse:.2f}")
        aiplatform.log_metrics({"rmse": rmse})

        model_filename = "model.joblib"
        joblib.dump(model, model_filename)
        print(f"   Model saved locally as {model_filename}")

        print("\n[5/6] Uploading and registering model to Vertex AI Model Registry...")

        # Prepare labels and description with Git metadata
        model_labels = {
            "git_commit_sha": git_commit_sha.lower() if git_commit_sha else "unknown",
            "git_author": git_author.lower().replace(" ", "-").replace("@", "-") if git_author else "unknown",
        }

        # Append full message to description (labels have length limits)
        full_model_description = f"{model_description}\nGit Commit: {git_commit_sha}\nAuthor: {git_author}\nMessage: {git_commit_message}"

        # Versioned GCS path for model artifacts
        versioned_model_artifact_uri = f"gs://{bucket_name}/model_artifacts/{git_commit_sha}/"

        uploaded_model = aiplatform.Model.upload(
            display_name=model_display_name,
            description=full_model_description,
            artifact_uri=versioned_model_artifact_uri,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
            labels=model_labels
        )

        storage_client = storage.Client(project=project_id)
        blob_path = f"model_artifacts/{git_commit_sha}/{model_filename}"
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(model_filename)
        print(f"   Model artifact uploaded to gs://{bucket_name}/{blob_path}")

        model_out.uri = uploaded_model.resource_name
        print(f"   Model registered: {model_out.uri}")
        model_out.metadata["resource_name"] = uploaded_model.resource_name
        model_out.metadata["rmse"] = float(rmse)

    print("\n COMPONENT 2 COMPLETE")
    print("="*70)

@component(
    base_image="gcr.io/deeplearning-platform-release/base-cpu:latest",
    packages_to_install=[
        "google-cloud-bigquery==3.11.4",
        "google-cloud-aiplatform>=1.38.0",
        "pandas==2.0.3",
        "scikit-learn==1.3.0",
        "joblib"
    ]
)
def batch_inference(
    project_id: str,
    dataset_name: str,
    table_name: str,
    model_uri: Input[Model]
):
    """Performs batch inference using the trained model."""
    import os
    import pandas as pd
    from google.cloud import bigquery, aiplatform
    import joblib

    print("="*70)
    print("COMPONENT 3: BATCH INFERENCE")
    print("="*70)

    print(f"\n[1/3] Initializing Vertex AI client...")
    aiplatform.init(project=project_id)

    print(f"\n[2/3] Loading model from: {model_uri.uri}")
    model_resource_name = model_uri.metadata["resource_name"]
    loaded_model = aiplatform.Model(model_resource_name=model_resource_name)
    print(f"   Model loaded: {loaded_model.display_name}")

    print(f"\n[3/3] Simulating batch inference on a small sample of data...")
    bq_client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset_name}.{table_name}"
    df_sample = bq_client.query(f"SELECT * FROM `{table_id}` LIMIT 10").to_dataframe()

    df_sample = pd.get_dummies(df_sample, columns=['ocean_proximity'], drop_first=True)
    df_sample = df_sample.dropna()
    X_inference = df_sample.drop('median_house_value', axis=1, errors='ignore')

    if not X_inference.empty:
        print(f"   Simulating batch prediction job for {len(X_inference)} instances.")
        print("   Batch inference simulated successfully.")
    else:
        print("   No data left for inference after preprocessing.")

    print("\n COMPONENT 3 COMPLETE")
    print("="*70)

@component(
    base_image="gcr.io/deeplearning-platform-release/base-cpu:latest",
    packages_to_install=[
        "google-cloud-aiplatform>=1.38.0"
    ]
)
def deploy_model_to_endpoint(
    project_id: str,
    location: str,
    model_display_name: str,
    model_uri: Input[Model],
    deployed_endpoint_out: Output[Model] # Using Model type for deployed endpoint output
):
    """Deploys the trained model to a Vertex AI Endpoint."""
    import os
    from google.cloud import aiplatform

    print("="*70)
    print("COMPONENT 4: DEPLOY MODEL TO ENDPOINT")
    print("="*70)

    print(f"\n[1/4] Initializing Vertex AI client...")
    aiplatform.init(project=project_id, location=location)

    print(f"\n[2/4] Loading model from: {model_uri.uri}")
    model_resource_name = model_uri.metadata["resource_name"]
    trained_model = aiplatform.Model(model_resource_name=model_resource_name)
    print(f"   Loaded model: {trained_model.display_name}")

    print(f"\n[3/4] Checking for existing or creating new endpoint: {model_display_name}")
    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{model_display_name}"', location=location)

    if endpoints:
        endpoint = endpoints[0]
        print(f"   Using existing endpoint: {endpoint.resource_name}")
    else:
        print(f"   Creating new endpoint: {model_display_name}")
        endpoint = aiplatform.Endpoint.create(
            display_name=model_display_name,
            project=project_id,
            location=location
        )
        print(f"   Created new endpoint: {endpoint.resource_name}")

    print(f"\n[4/4] Deploying model '{trained_model.display_name}' to endpoint '{endpoint.display_name}'...")
    traffic_split = {"0": 100} # All traffic to the newly deployed model version
    deployed_model = endpoint.deploy(
        model=trained_model,
        deployed_model_display_name=f"{model_display_name}-deployed",
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=1,
        traffic_split=traffic_split
    )
    print(f"   Model deployed. Deployed Model ID: {deployed_model.id}")

    deployed_endpoint_out.uri = endpoint.resource_name
    deployed_endpoint_out.metadata["endpoint_resource_name"] = endpoint.resource_name
    deployed_endpoint_out.metadata["deployed_model_id"] = deployed_model.id
    print(f"   Deployed Endpoint URI: {deployed_endpoint_out.uri}")

    print("\n COMPONENT 4 COMPLETE")
    print("="*70)


# ============================================================================
# PIPELINE DEFINITION
# ============================================================================

@dsl.pipeline(
    name="california-housing-mlops-complete", # Hardcode name for simplicity in CI/CD
    description="Complete MLOps pipeline - Simplified version"
)
def california_housing_pipeline(
    project_id: str,
    location: str,
    bucket_name: str,
    dataset_name: str,
    table_name: str,
    model_display_name: str,
    model_description: str,
    git_commit_sha: str = '', # Existing pipeline parameter
    git_author: str = '',
    git_commit_message: str = ''
):
    """Complete MLOps Pipeline"""

    setup_task = setup_and_prepare_data(
        project_id=project_id,
        location=location,
        bucket_name=bucket_name,
        dataset_name=dataset_name,
        table_name=table_name,
        git_commit_sha=git_commit_sha # Pass Git commit SHA to setup_and_prepare_data
    )

    train_task = train_and_register_model(
        project_id=project_id,
        location=location,
        dataset_name=dataset_name,
        table_name=table_name,
        bucket_name=bucket_name,
        model_display_name=model_display_name,
        model_description=model_description,
        git_commit_sha=git_commit_sha,
        git_author=git_author,
        git_commit_message=git_commit_message,
        dataset_in=setup_task.outputs['dataset_out']
    )

    deploy_task = deploy_model_to_endpoint(
        project_id=project_id,
        location=location,
        model_display_name=model_display_name,
        model_uri=train_task.outputs['model_out'] # Use the output model from training
    )

    inference_task = batch_inference(
        project_id=project_id,
        dataset_name=dataset_name,
        table_name=table_name,
        model_uri=train_task.outputs['model_out'] # Use the output model from training
    )

# ============================================================================
# PIPELINE EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Get configuration from environment variables
    PROJECT_ID = os.environ.get("PROJECT_ID", "bits-mlops-assignment1")
    LOCATION = os.environ.get("LOCATION", "us-central1")
    BUCKET_NAME = os.environ.get("BUCKET_NAME", "mlops_bucket_assignment_01")
    DATASET_NAME = os.environ.get("DATASET_NAME", "mlops_datas")
    TABLE_NAME = os.environ.get("TABLE_NAME", "california")

    MODEL_DISPLAY_NAME = os.environ.get("MODEL_DISPLAY_NAME", "california-housing-model")
    MODEL_DESCRIPTION = os.environ.get("MODEL_DESCRIPTION", "Random Forest model for California housing price prediction")

    # Retrieve Git metadata from environment variables
    GIT_COMMIT_SHA = os.environ.get("COMMIT_SHA", "")
    GIT_AUTHOR = os.environ.get("COMMIT_AUTHOR", "")
    GIT_COMMIT_MESSAGE = os.environ.get("COMMIT_MESSAGE", "")

    PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root"
    PIPELINE_NAME = "california-housing-mlops-complete" # Also hardcode here to match @dsl.pipeline decorator

    print("\n" + "="*80)
    print("COMPILING PIPELINE")
    print("="*80)

    pipeline_file = "housing_pipeline_simplified.json"

    compiler.Compiler().compile(
        pipeline_func=california_housing_pipeline,
        package_path=pipeline_file
    )
    print(f" Pipeline compiled: {pipeline_file}")

    print("\n" + "="*80)
    print("SUBMITTING PIPELINE")
    print("="*80)

    aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=PIPELINE_ROOT)

    job_id = f"housing-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    job = aiplatform.PipelineJob(
        display_name=job_id,
        template_path=pipeline_file,
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "project_id": PROJECT_ID,
            "location": LOCATION,
            "bucket_name": BUCKET_NAME,
            "dataset_name": DATASET_NAME,
            "table_name": TABLE_NAME,
            "model_display_name": MODEL_DISPLAY_NAME,
            "model_description": MODEL_DESCRIPTION,
            "git_commit_sha": GIT_COMMIT_SHA,
            "git_author": GIT_AUTHOR,
            "git_commit_message": GIT_COMMIT_MESSAGE
        },
        enable_caching=False
    )

    job.submit()
    print(f" Submitted: {job_id}")
    print(f"\nConsole: {job._dashboard_uri()}")
    print("\nWaiting for completion...")
    job.wait()
    print("\n" + "="*80)
    print(" PIPELINE COMPLETED!")
    print("="*80)
