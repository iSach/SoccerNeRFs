"""
Some experiments with NeRFPlayer on Synth Paderborn dataset (synthetic player).

Related to Experiment 5.5.1.
"""
from ns_experiment import Experiment


def main():
    """
    Main fct.
    """
    exp = Experiment(
        name="np_synthpaderborn",
        cam_path="/workspace/data/synth_paderborn/camera_paths/around_player.json",
        model="nerfplayer",
        dataset="synthpaderborn-data",
    )
    exp.set_params(
        [
            {
                "pipeline.datamanager.isg": False,
            },
            {
                "pipeline.datamanager.isg": True,
            },
            {
                "pipeline.datamanager.isg": False,
            },
        ], 
        [
            {
                "depth_mask": "none",
                "cam_split_setup": "low",
                "scene_scale": 0.85,
            },
            {
                "depth_mask": "none",
                "cam_split_setup": "low",
                "scene_scale": 0.85,
            },
            {
                "depth_mask": "mask",
                "cam_split_setup": "real",
                "scene_scale": 0.85,
            }
        ])
    exp.run(do_eval=False)

if __name__ == "__main__":
    main()