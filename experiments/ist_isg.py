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
        name="ist_isg",
        cam_path="/workspace/data/synth_paderborn/camera_paths/around_player.json",
        model="k-planes",
        dataset="synthpaderborn-data",
    )
    exp.set_params(
        [
            {
                "pipeline.datamanager.is-pixel-ratio": 0.15, 
                "pipeline.datamanager.isg": False,
                "pipeline.datamanager.ist-range": 1.0,
                "pipeline.datamanager.iters-to-start-is": 3500,
            },
            {
                "pipeline.datamanager.is-pixel-ratio": 0.3, 
                "pipeline.datamanager.isg": False,
                "pipeline.datamanager.ist-range": 0.5,
                "pipeline.datamanager.iters-to-start-is": 3500,
            },
            {
                "pipeline.datamanager.is-pixel-ratio": 0.3, 
                "pipeline.datamanager.isg": False,
                "pipeline.datamanager.ist-range": 0.75,
                "pipeline.datamanager.iters-to-start-is": 3500,
            },
            {
                "pipeline.datamanager.is-pixel-ratio": 0.3, 
                "pipeline.datamanager.isg": False,
                "pipeline.datamanager.ist-range": 1.0,
                "pipeline.datamanager.iters-to-start-is": 3500,
            },
            {
                "pipeline.datamanager.is-pixel-ratio": 0.65, 
                "pipeline.datamanager.isg": False,
                "pipeline.datamanager.ist-range": 0.5,
                "pipeline.datamanager.iters-to-start-is": 3500,
            },
            {
                "pipeline.datamanager.is-pixel-ratio": 0.05, 
                "pipeline.datamanager.isg": False,
                "pipeline.datamanager.ist-range": 0.25,
                "pipeline.datamanager.iters-to-start-is": 3500,
            },
            {
                "pipeline.datamanager.is-pixel-ratio": 0.15, 
                "pipeline.datamanager.isg": False,
                "pipeline.datamanager.ist-range": 0.25,
                "pipeline.datamanager.iters-to-start-is": 3500,
            },
            {
                "pipeline.datamanager.is-pixel-ratio": 0.25, 
                "pipeline.datamanager.isg": False,
                "pipeline.datamanager.ist-range": 0.25,
                "pipeline.datamanager.iters-to-start-is": 3500,
            },
            {
                "pipeline.datamanager.is-pixel-ratio": 0.15, 
                "pipeline.datamanager.isg": False,
                "pipeline.datamanager.ist-range": 0.05,
                "pipeline.datamanager.iters-to-start-is": 3500,
            },
            {
                "pipeline.datamanager.is-pixel-ratio": 0.15, 
                "pipeline.datamanager.isg": False,
                "pipeline.datamanager.ist-range": 0.5,
                "pipeline.datamanager.iters-to-start-is": 3500,
            },
            {
                "pipeline.datamanager.is-pixel-ratio": 0.05, 
                "pipeline.datamanager.isg": True,
                "pipeline.datamanager.ist-range": 0.25,
                "pipeline.datamanager.iters-to-start-is": 3500,
            },
            {
                "pipeline.datamanager.is-pixel-ratio": 0.15, 
                "pipeline.datamanager.isg": True,
                "pipeline.datamanager.ist-range": 0.25,
                "pipeline.datamanager.iters-to-start-is": 3500,
            },
        ], None)
    exp.run(do_eval=False)

if __name__ == "__main__":
    main()
