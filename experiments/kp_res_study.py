"""
Studies the resolution of K-Planes (Stadium).

Experiment 5.3.5.
"""
from ns_experiment import Experiment


def main():
    """
    Main fct.
    """
    exp = Experiment(
        name="kp_res_study",
        cam_path="data/stadium/camera_paths/nicecam.json",
        model="k-planes",
        dataset="stadium-data",
    )
    exp.set_params(
        [
            {
                "pipeline.model.multiscale-res": "1 2 4 8",
                "pipeline.model.feature-dim": "32",
                "pipeline.model.spacetime-resolution": "64 64 64 100"  # Base res.
            },
            {
                "pipeline.model.multiscale-res": "1 2 4 8 16",
                "pipeline.model.feature-dim": "32",
                "pipeline.model.spacetime-resolution": "64 64 64 100"
            },
            {
                "pipeline.model.multiscale-res": "1 2 4 8 16 32",
                "pipeline.model.feature-dim": "24",
                "pipeline.model.spacetime-resolution": "64 64 64 100"
            },
        ], None)
    exp.run(do_eval=True)

if __name__ == "__main__":
    main()