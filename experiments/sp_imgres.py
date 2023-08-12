"""
Image Resolution on Synth Paderborn.

Not in report.
"""

import ns_experiment


def main():
    """
    Main fct.
    """
    exp = ns_experiment.Experiment(
        name="sp_imgres_x1",
        cam_path="/workspace/data/synth_paderborn/camera_paths/around_player_4_5_scaled.json",
        model="k-planes",
        dataset="synthpaderborn-data",
    )
    exp.set_params(
        [
            {
                "downscale-factor": 2,
                "fps-downsample": 5,
            },
        ], 
        [
            {
                "downscale-factor": 1,
                "fps-downsample": 5,
            },
        ])
    exp.run(do_eval=True)

if __name__ == "__main__":
    main()
