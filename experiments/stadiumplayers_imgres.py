"""
Comparison of different image resolutions for K-Planes on Stadium Players

Experiment 5.4.2.
"""

import ns_experiment


def main():
    """
    Main fct.
    """
    exp = ns_experiment.Experiment(
        name="stadiumplayers_imgres",
        cam_path=[
            "/workspace/data/stadium_players/camera_paths/center.json",
            "/workspace/data/stadium_players/camera_paths/fallingball.json",
            "/workspace/data/stadium_players/camera_paths/goalright.json",
            "/workspace/data/stadium_players/camera_paths/playerleft.json",
            "/workspace/data/stadium_players/camera_paths/playerright.json",
            "/workspace/data/stadium_players/camera_paths/shooter.json",
            "/workspace/data/stadium_players/camera_paths/global.json",
        ],
        model="k-planes",
        dataset="stadiumplayers-data",
    )
    exp.set_params(
        [
            {
                "pipeline.datamanager.train-num-rays-per-batch": 8192,
                "pipeline.model.feature-dim": 16,
                "pipeline.datamanager.train-num-images-to-sample-from": 500,
                "pipeline.datamanager.train-num-times-to-repeat-images": 1000,
                "steps-per-eval-batch": 0,
            },
            {
                "pipeline.datamanager.train-num-rays-per-batch": 4096,
                "pipeline.model.feature-dim": 32,
                "pipeline.datamanager.train-num-images-to-sample-from": 500,
                "pipeline.datamanager.train-num-times-to-repeat-images": 1000,
                "steps-per-eval-batch": 0, 
            },
            # {
            #     "pipeline.datamanager.train-num-rays-per-batch": 8192,
            #     "pipeline.model.feature-dim": 32,
            #     "pipeline.datamanager.train-num-images-to-sample-from": 1000,
            #     "pipeline.datamanager.train-num-times-to-repeat-images": 2000,
            #     "steps-per-eval-batch": 0,
            # },
            # {
            #     "pipeline.datamanager.train-num-rays-per-batch": 8192,
            #     "pipeline.model.feature-dim": 32,
            #     "pipeline.datamanager.train-num-images-to-sample-from": 1000,
            #     "pipeline.datamanager.train-num-times-to-repeat-images": 2000,
            #     "steps-per-eval-batch": 0,
            # },
        ],
        [
            {
                "downscale-factor": 1,
            },
            {
                "downscale-factor": 1,
            },
            # {
            #     "downscale-factor": 2,
            # },
            # {
            #     "downscale-factor": 4,
            # }
        ]
    )
    exp.run(do_eval=True)

if __name__ == "__main__":
    main()
