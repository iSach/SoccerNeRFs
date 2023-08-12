"""
Synthetic Balls: Size for both K-Planes and NeRFPlayer.

Experiment 5.3.5.
"""

import ns_experiment


def main():
    """
    Main fct.
    """
    exp = ns_experiment.Experiment(
        name="modelsize_kplanes",
        cam_path="/workspace/data/stadium/camera_paths/nicecam.json",
        model="k-planes",
        dataset="stadium-data",
    )
    exp.set_params(
        [
            {
                "pipeline.model.multiscale-res": "1 2 4 8",
                "pipeline.model.feature-dim": 32,
            },
            {
                "pipeline.model.multiscale-res": "1 2 4 8 16",
                "pipeline.model.feature-dim": 32,
            },
            {
                "pipeline.model.multiscale-res": "1 2 4 8 16 32",
                "pipeline.model.feature-dim": 24,
            },
        ],
        None
    )
    exp.run(do_eval=True)
    
    exp = ns_experiment.Experiment(
        name="modelsize_npnerfacto",
        cam_path="/workspace/data/stadium/camera_paths/nicecam.json",
        model="nerfplayer-nerfacto",
        dataset="stadium-data",
    )
    exp.set_params(
        [
            {
                "pipeline.model.log2-hashmap-size": 19, 
            },
            {
                "pipeline.model.temporal-dim": 24,
                "pipeline.model.log2-hashmap-size": 20, 
            },
        ],
        None
    )
    exp.run(do_eval=True)

if __name__ == "__main__":
    main()
