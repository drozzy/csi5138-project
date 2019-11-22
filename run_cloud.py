import subprocess
from datetime import datetime
import plac
import os
MAX_EPOCHS = 100

PROJECT_ID="andriy"
BUCKET_NAME="chumak"
REGION = "us-central1"

IMAGE_REPO_NAME="csi5138-project"
IMAGE_TAG="gpu"
IMAGE_URI = f'gcr.io/{PROJECT_ID}/{IMAGE_REPO_NAME}:{IMAGE_TAG}'

def build_and_push():
    print(IMAGE_URI)
    subprocess.run(["docker", "build", "-f", "Dockerfile", "-t", IMAGE_URI, "./"])
    subprocess.run(["docker", "push", IMAGE_URI])

def main(job_only: ('Skip and docker building and pushing', 'flag')): 
    if not job_only:
        build_and_push()

    n = datetime.now()
    timestamp = f'{n.year}_{n.month}_{n.day}__{n.hour}_{n.minute}_{n.second}'
    JOB_NAME= f'csi5138_job_{timestamp}'
    STUDY = f'study_{timestamp}'

    PROJECT_DIR = f'gs://{BUCKET_NAME}/csi5138_project'
    STUDY_DIR = os.path.join(PROJECT_DIR, f"studies/{STUDY}")
    DATA_DIR = os.path.join(PROJECT_DIR, "data")

    RESULTS_DIR = os.path.join(STUDY_DIR, "results")
    MODELS_DIR  = os.path.join(STUDY_DIR, "models")

    subprocess.run(["gcloud", "ai-platform", "jobs", "submit", "training", JOB_NAME,
        "--scale-tier", "BASIC_GPU",
        "--region", REGION,
        "--master-image-uri", IMAGE_URI,
        "--",
        RESULTS_DIR,
        MODELS_DIR,
        DATA_DIR,
        f'{MAX_EPOCHS}'])

    subprocess.run(["gcloud", "ai-platform", "jobs", "describe", JOB_NAME])
    subprocess.run(["gcloud", "ai-platform", "jobs", "stream-logs", JOB_NAME])

if __name__ == '__main__':
    plac.call(main)