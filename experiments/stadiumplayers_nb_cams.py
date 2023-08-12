"""
Comparison of different numbers of cameras for K-Planes on Stadium Players

Experiment 5.4.3.
"""

import ns_experiment


def main():
    """
    Main fct.
    """
    exp = ns_experiment.Experiment(
        name="stadiumplayers_nb_cams",
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
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
        ],
        [
            {
                "nb-train-cameras": 110,
            },
            {
                "nb-train-cameras": 70,
            },
            {
                "nb-train-cameras": 40,
            },
            {
                "nb-train-cameras": 30,
            },
            {
                "nb-train-cameras": 25,
            },
            {
                "nb-train-cameras": 20,
            },
            {
                "nb-train-cameras": 15,
            },
            {
                "nb-train-cameras": 10,
            },
            {
                "nb-train-cameras": 7
            },
            {
                "nb-train-cameras": 5,
            },
            {
                "nb-train-cameras": 3,
            },
        ]
    )
    exp.run(do_eval=False)

if __name__ == "__main__":
    main()
