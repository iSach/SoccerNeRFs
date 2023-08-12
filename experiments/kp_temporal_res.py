"""
K-Planes temporal resolution experiment, on Paderborn.

Not present in report. 

This was not really discussed, but it seems that even low resolutions (e.g., 25) capture the movement pretty well.
"""
from ns_experiment import Experiment


def main():
    """
    Main fct.
    """
    exp = Experiment(
        name="kp_temporal_res",
        cam_path="data/paderborn/camera_paths/bounded_smallbox540.json",
        model="k-planes",
        dataset="paderborn-data",
    )
    exp.set_params(
        [
            {
                "pipeline.model.space-time-resolution": "64 64 64 10",
            },
            {
                "pipeline.model.space-time-resolution": "64 64 64 25",
            },
            {
                "pipeline.model.space-time-resolution": "64 64 64 50",
            },
            {
                "pipeline.model.space-time-resolution": "64 64 64 75",
            },
            {
                "pipeline.model.space-time-resolution": "64 64 64 100",
            },
            {
                "pipeline.model.space-time-resolution": "64 64 64 125",
            },
            {
                "pipeline.model.space-time-resolution": "64 64 64 150",
            },
            {
                "pipeline.model.space-time-resolution": "64 64 64 200",
            },
            {
                "pipeline.model.space-time-resolution": "64 64 64 250",
            },
            {
                "pipeline.model.space-time-resolution": "64 64 64 400",
            },
        ], 
        [
            {
                "scene-scale": 0.7,
                "depth-maps-to-use": "mask",
            },
            {
                "scene-scale": 0.7,
                "depth-maps-to-use": "mask",
            },
            {
                "scene-scale": 0.7,
                "depth-maps-to-use": "mask",
            },
            {
                "scene-scale": 0.7,
                "depth-maps-to-use": "mask",
            },
            {
                "scene-scale": 0.7,
                "depth-maps-to-use": "mask",
            },
            {
                "scene-scale": 0.7,
                "depth-maps-to-use": "mask",
            },
            {
                "scene-scale": 0.7,
                "depth-maps-to-use": "mask",
            },
            {
                "scene-scale": 0.7,
                "depth-maps-to-use": "mask",
            },
            {
                "scene-scale": 0.7,
                "depth-maps-to-use": "mask",
            },
            {
                "scene-scale": 0.7,
                "depth-maps-to-use": "mask",
            },
        ])
    exp.run(do_eval=False)

if __name__ == "__main__":
    main()
