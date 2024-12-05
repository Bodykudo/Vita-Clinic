from fastapi import APIRouter, Depends

from src.auth.dependencies import ApiKeyHeader
from src.inference.schema import InferenceSchema
from src.inference.celery_jobs import predict_mammography_task

mammography_cancer_classification_router = APIRouter()


@mammography_cancer_classification_router.post(
    "/",
    dependencies=[Depends(ApiKeyHeader())],
    summary="Submit a mammography prediction task",
    description="Submit a DICOM image for mammography cancer prediction",
)
async def submit_prediction(inferenceSchema: InferenceSchema):
    """
    Submit a mammography cancer prediction task.

    Args:
        inferenceSchema (InferenceSchema): Schema containing the instance URL of the DICOM image.

    Returns:
        dict: Contains the task ID of the submitted Celery task.
    """
    task = predict_mammography_task.delay(
        inferenceSchema.predictionId, inferenceSchema.instance
    )
    return {"task_id": task.id}
