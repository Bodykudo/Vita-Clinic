from fastapi import APIRouter

from src.inference.brain_tumors_classification.routes import (
    brain_tumors_classification_router,
)
from src.inference.chest_ct_cancer_classification.routes import (
    chest_ct_cancer_classification_router,
)
from src.inference.mammography_cancer_classification.routes import (
    mammography_cancer_classification_router,
)

inference_router = APIRouter()

inference_router.include_router(
    brain_tumors_classification_router, prefix="/brain-tumors-classification"
)
inference_router.include_router(
    chest_ct_cancer_classification_router, prefix="/chest-ct-cancer-classification"
)
inference_router.include_router(
    mammography_cancer_classification_router, prefix="/mammography-cancer-classification",
)
