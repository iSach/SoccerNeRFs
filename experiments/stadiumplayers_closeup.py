"""
Close up training.

Experiment 5.4.4.
"""

import ns_experiment


def main():
    """
    Main fct.
    """
    exp = ns_experiment.Experiment(
        name="stadiumplayers_closeup",
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
            },
            {
            },
            {
            },
            {
            },
        ],
        [
            {
                "closeup-training": False,
                "nb-train-cameras": 110,
            },
            {
                "closeup-training": False,
                "nb-train-cameras": 25,
            },
            {
                "closeup-training": True,
                "nb-train-cameras": 110,
            },
            {
                "closeup-training": True,
                "nb-train-cameras": 25,
            },
        ]
    )
    exp.run(do_eval=True)

if __name__ == "__main__":
    main()
