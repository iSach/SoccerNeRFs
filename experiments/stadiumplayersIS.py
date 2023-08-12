"""
Comparison of different importance sampling parameters for K-Planes on Synthetic Players.

Experiment 5.4.1 (End of it).
"""

import ns_experiment


def main():
    """
    Main fct.
    """
    exp = ns_experiment.Experiment(
        name="stadiumplayers_is",
        cam_path=[
            "/workspace/data/stadium_players/camera_paths/center.json",
            "/workspace/data/stadium_players/camera_paths/fallingball.json",
            "/workspace/data/stadium_players/camera_paths/goalright.json",
            "/workspace/data/stadium_players/camera_paths/playerleft.json",
            "/workspace/data/stadium_players/camera_paths/playerright.json",
            "/workspace/data/stadium_players/camera_paths/shooter.json",
        ],
        model="k-planes",
        dataset="stadiumplayers-data",
    )
    exp.set_params(
        [
            {
                "pipeline.datamanager.is-pixel-ratio": 0.05, 
                "pipeline.datamanager.isg": True,
                "pipeline.datamanager.ist-range": 0.25,
                "pipeline.model.multiscale-res": "1 2 4 8 16",
                "pipeline.model.feature-dim": 32,
            },
            {
                "pipeline.datamanager.is-pixel-ratio": 0.15, 
                "pipeline.datamanager.isg": False,
                "pipeline.datamanager.ist-range": 0.25,
                "pipeline.model.multiscale-res": "1 2 4 8 16",
                "pipeline.model.feature-dim": 32,
            },
            {
                "pipeline.datamanager.is-pixel-ratio": 0.05, 
                "pipeline.datamanager.isg": True,
                "pipeline.datamanager.ist-range": 0.25,
                "pipeline.model.multiscale-res": "1 2 4 8",
                "pipeline.model.feature-dim": 48,
            },
            {
                "pipeline.datamanager.is-pixel-ratio": 0.15, 
                "pipeline.datamanager.isg": False,
                "pipeline.datamanager.ist-range": 0.25,
                "pipeline.model.multiscale-res": "1 2 4 8",
                "pipeline.model.feature-dim": 48,
            },
        ], None)
    exp.run(do_eval=True)

if __name__ == "__main__":
    main()
