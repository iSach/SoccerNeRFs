"""
Runs of KP and NP with Importance Sampling (IS).

Experiment 5.3.2.
"""

import ns_experiment


def main():
    """
    Main fct.
    """
    exp = ns_experiment.Experiment(
        name="stadium_is_kp",
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
        ],
        None
    )
    exp.run(do_eval=False)
    
    exp = ns_experiment.Experiment(
        name="stadium_is_npnerfacto",
        cam_path=[
            "/workspace/data/stadium/camera_paths/nicecam540.json",
            "/workspace/data/stadium/camera_paths/nicecam.json",
        ],
        model="nerfplayer-nerfacto",
        dataset="stadium-data",
    )
    exp.set_params(
        [
            {
                "pipeline.datamanager.use-importance-sampling": True, 
            },
        ],
        None
    )
    exp.run(do_eval=False)

if __name__ == "__main__":
    main()
