from celery import Celery
import asyncio
import httpx
import torch

from src.dicom.service import DicomService
from src.inference.brain_tumors_classification.service import (
    BrainTumorClassificationService,
)
from src.inference.chest_ct_cancer_classification.service import (
    ChestCTCancerClassificationService,
)
from src.inference.mammography_cancer_classification.service import (
    MammographyInferenceService
)

from src.config import Config

torch.set_num_threads(1)

celery_app = Celery(
    "tasks",
    broker=Config.CELERY_BROKER_URL,
    backend=Config.CELERY_RESULT_BACKEND,
)

dicom_service = DicomService()
brain_tumor_service = None
ct_scan_service = None
mammography_service = None


async def update_prediction_result(
    prediction_id: str, prediction: str, probability: float
):
    """
    Update the prediction result in the backend.

    Args:
        prediction_id (str): The prediction ID.
        prediction (str): The prediction class.
        probability (float): The prediction probability.

    Returns:
        None
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.patch(
                f"{Config.BACKEND_URL}/cdss/{prediction_id}",
                json={"result": prediction, "probability": probability},
                headers={"x-api-key": Config.API_KEY},
            )
            response.raise_for_status()
        except httpx.RequestError as e:
            print(f"Request failed: {e}")
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")


async def fail_prediction(prediction_id: str):
    """
    Update the prediction result in the backend.

    Args:
        prediction_id (str): The prediction ID.

    Returns:
        None
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.patch(
                f"{Config.BACKEND_URL}/cdss/{prediction_id}/fail",
                headers={"x-api-key": Config.API_KEY},
            )
            response.raise_for_status()
        except httpx.RequestError as e:
            print(f"Request failed: {e}")
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")


@celery_app.task
def predict_brain_tumors_task(prediction_id: str, instance_url: str):
    """
    Celery task for predicting brain tumor type.

    Args:
        instance_url (str): The DICOM file instance URL.

    Returns:
        dict: Prediction result with class and probability.
    """
    try:
        loop = asyncio.get_event_loop()
        image_bytes = loop.run_until_complete(
            dicom_service.convert_dicom_to_image(instance_url)
        )

        global brain_tumor_service
        if brain_tumor_service is None:
            brain_tumor_service = BrainTumorClassificationService(
                "./src/inference/brain_tumors_classification/weights/brain_tumors_classification.pt"
            )

        prediction = brain_tumor_service.predict(image_bytes.getvalue())
        loop.run_until_complete(
            update_prediction_result(
                prediction_id, prediction["prediction"], prediction["probability"]
            )
        )

        return prediction
    except Exception as e:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(fail_prediction(prediction_id))
        raise e


@celery_app.task
def predict_chest_ctscan_task(prediction_id: str, instance_url: str):
    """
    Celery task for predicting chest CT cancer type.

    Args:
        instance_url (str): The DICOM file instance URL.

    Returns:
        dict: Prediction result with class and probability.
    """
    try:
        loop = asyncio.get_event_loop()
        image_bytes = loop.run_until_complete(
            dicom_service.convert_dicom_to_image(instance_url)
        )

        global ct_scan_service
        if ct_scan_service is None:
            ct_scan_service = ChestCTCancerClassificationService(
                "./src/inference/chest_ct_cancer_classification/weights/chest_ct_cancer_classification.pt"
            )

        prediction = ct_scan_service.predict(image_bytes.getvalue())
        loop.run_until_complete(
            update_prediction_result(
                prediction_id, prediction["class"], prediction["probability"]
            )
        )

        return prediction
    except Exception as e:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(fail_prediction(prediction_id))
        raise e

@celery_app.task
def predict_mammography_task(prediction_id: str, instance_url: str):
    """
    Celery task for predicting mammography cancer type.

    Args:
        prediction_id (str): The unique ID for the prediction task.
        instance_url (str): The DICOM file instance URL.

    Returns:
        dict: Prediction result with class and probability.
    """
    try:
        # Get the asyncio event loop
        loop = asyncio.get_event_loop()

        # Convert the DICOM file to an image (returns image bytes)
        image_bytes = loop.run_until_complete(
            dicom_service.convert_dicom_to_image(instance_url)
        )

        # Load the global mammography inference service if not already loaded
        global mammography_service
        if mammography_service is None:
            mammography_service = MammographyInferenceService(
                "./src/inference/mammography_cancer_classification/weights/mammography_cancer_classification.pt"      
            )

        prediction = mammography_service.predict(image_bytes.getvalue())

        loop.run_until_complete(
            update_prediction_result(
                prediction_id, prediction["class"], prediction["probability"]
            )
        )

     
        return prediction

    except Exception as e:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(fail_prediction(prediction_id))
        raise e