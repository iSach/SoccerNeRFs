"""
Comparison of different importance sampling parameters for K-Planes on Synthetic paderborn.

Experiment 5.5.5.
"""

import ns_experiment


def main():
    """
    Main fct.
    """
    exp = ns_experiment.Experiment(
        name="sp_kp_isg",
        cam_path="/workspace/data/synth_paderborn/camera_paths/around_player_4_5_scaled.json",
        model="k-planes",
        dataset="synthpaderborn-data",
    )
    exp.set_params(
        [
            {
                "pipeline.datamanager.isg-gamma": 1e-2, 
            },
            {
                "pipeline.datamanager.isg-gamma": 1e-4, 
            },
            {
                "pipeline.datamanager.isg-gamma": 1e-3, 
            },
            {
                "pipeline.datamanager.isg-gamma": 1e-1, 
            },
        ], None)
    exp.run(do_eval=True)

if __name__ == "__main__":
    main()
