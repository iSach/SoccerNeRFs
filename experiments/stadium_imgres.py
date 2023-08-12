"""
Comparison of different image resolutions for K-Planes on Synthetic Balls.

Experiment 5.3.4.
"""

import ns_experiment


def main():
    """
    Main fct.
    """
    exp = ns_experiment.Experiment(
        name="stadium_imgres",
        cam_path=[
            "/workspace/data/stadium/camera_paths/nicecam540.json",
            "/workspace/data/stadium/camera_paths/nicecam.json",
        ],
        model="k-planes",
        dataset="stadium-data",
    )
    exp.set_params(
        [
            {
                "pipeline.datamanager.use-importance-sampling": True, 
            },
            {
                "pipeline.datamanager.use-importance-sampling": True, 
            },
            {
                "pipeline.datamanager.use-importance-sampling": True, 
            },
            {
                "pipeline.datamanager.use-importance-sampling": True, 
            },
        ],
        [
            {
                "downscale-factor": 1,
            },
            {
                "downscale-factor": 2,
            },
            {
                "downscale-factor": 4,
            },
            {
                "downscale-factor": 8,
            }
        ]
    )
    exp.run(do_eval=False)

if __name__ == "__main__":
    main()
