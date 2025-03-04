import numpy as np
import json
import os

from inference_PCAA import VARIATION, CGAAE_inference
import constants

if __name__ == "__main__":

    model_names = [
        "PCAA_AblV4.2.1",
        "PCAA_AblV4.2.2",
        "PCAA_AblV4.2.3",
        "PCAA_AblV4.2.4",
        "PCAA_AblV4.2.5",
        "PCAA_AblV4.4.1",
        "PCAA_AblV4.4.2",
        "PCAA_AblV4.4.3",
        "PCAA_AblV4.4.4",
        "PCAA_AblV4.4.5",
        "PCAA_AblV4.6.1",
        "PCAA_AblV4.6.2",
        "PCAA_AblV4.6.3",
        "PCAA_AblV4.6.4",
        "PCAA_AblV4.6.5",
        "PCAA_AblV4.8.1",
        "PCAA_AblV4.8.2",
        "PCAA_AblV4.8.3",
        "PCAA_AblV4.8.4",
        "PCAA_AblV4.8.5",
    ]

    ks = [6, 4, 2, 1]

    inference_scenarios = [[constants.SCENARIO.FREE_WALK], [
        constants.SCENARIO.HANDS_IN_POCKETS], [constants.SCENARIO.SMARTPHONE]]

    for scenarios in inference_scenarios:

        CGAAE_inference(model_names=model_names, ks=ks,
                        scenarios_list=scenarios)
