import matplotlib.pyplot as plt
from . import HazeRemover
from .benchmarks_utils import EvaluationMetricManager, evaluate_haze_remover


# Parameter to limit the haze removal
def evaluate_impact_of_omega(omegas, image, save_folder):
    for omega in omegas:
        with EvaluationMetricManager("OMEGA", omega):
            haze_remover = HazeRemover(
                image,
                use_soft_matting=False,
                guided_image_filtering=True,
                omega=omega
            )
            evaluate_haze_remover(haze_remover, save_folder, "OMEGA", omega)


# Parameter to threshold minimal values of transmission map when computing radiance
def evaluate_impact_of_t0(t0s, image, save_folder):
    for t0 in t0s:
        with EvaluationMetricManager("T0", t0):
            haze_remover = HazeRemover(
                image,
                use_soft_matting=False,
                guided_image_filtering=True,
                t0=t0
            )
            evaluate_haze_remover(haze_remover, save_folder, "t0", t0)


# Patch size to compute transmission
def evaluate_impact_of_patch_size(patch_sizes, image, save_folder):
    for patch_size in patch_sizes:
        with EvaluationMetricManager("PATCH SIZE", patch_size):
            haze_remover = HazeRemover(
                image,
                use_soft_matting=False,
                guided_image_filtering=True,
                patch_size=patch_size
            )
            evaluate_haze_remover(haze_remover, save_folder, "patch_size", patch_size)


# Type of matting performed
def evaluate_impact_of_matting_methods(image, save_folder):
    with EvaluationMetricManager("Matting", "None"):
        haze_remover = HazeRemover(
            image,
            use_soft_matting=False,
            guided_image_filtering=False,
        )
        evaluate_haze_remover(haze_remover, save_folder, "matting", "none")

    with EvaluationMetricManager("Matting", "Soft matting"):
        haze_remover = HazeRemover(
            image,
            use_soft_matting=True,
            guided_image_filtering=False,
        )
        evaluate_haze_remover(haze_remover, save_folder, "matting", "soft")

    with EvaluationMetricManager("Matting", "Guided filtering"):
        haze_remover = HazeRemover(
            image,
            use_soft_matting=False,
            guided_image_filtering=True,
        )
        evaluate_haze_remover(haze_remover, save_folder, "matting", "guided")
