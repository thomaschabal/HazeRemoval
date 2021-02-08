import os
import matplotlib.pyplot as plt
from .constants import GAMMA


class EvaluationMetricManager:
    def __init__(self, metric_name, metric_value):
        self.metric_name = metric_name
        self.metric_value = metric_value
        
    def __enter__(self):
        print('\n', "=" * 30, '\n', f"Evaluating {self.metric_name} = {self.metric_value}")
        
    def __exit__(self, exc_t, exc_v, trace):
        None


def save_radiance_transmission(save_folder, radiance, transmission, image, metric_name, metric_value):
    if not os.path.exists(save_folder + "original.jpg"):
        plt.imsave(save_folder + "original.jpg", image)
    plt.imsave(save_folder + f"radiance_{metric_name}_{metric_value}.jpg", radiance)
    plt.imsave(save_folder + f"transmission_{metric_name}_{metric_value}.jpg", transmission, cmap='gray')


def evaluate_haze_remover(haze_remover, save_folder, metric_name, metric_value, gamma=GAMMA):
    radiance, transmission, _ = haze_remover.remove_haze(gamma)
    save_radiance_transmission(save_folder, radiance, transmission, haze_remover.image, metric_name, metric_value)
