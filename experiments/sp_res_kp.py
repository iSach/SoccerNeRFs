"""
Some trails at different settings on synthetic paderborn data for K-Planes.

Experiment 5.5.1.
"""
from ns_experiment import Experiment


def main():
    """
    Main fct.
    """
    exp = Experiment(
        name="sp_res_kp",
        cam_path="/workspace/data/synth_paderborn/camera_paths/around_player.json",
        model="k-planes",
        dataset="synthpaderborn-data",
    )
    exp.set_params(
        [
            {
                "pipeline.model.multiscale-res": "1", 
                "pipeline.model.spacetime-resolution": "64 64 64 100", 
            },
            {
                "pipeline.model.multiscale-res": "1 2", 
                "pipeline.model.spacetime-resolution": "64 64 64 100", 
            },
            {
                "pipeline.model.multiscale-res": "1 2 4", 
                "pipeline.model.spacetime-resolution": "64 64 64 100", 
            },
            {
                "pipeline.model.multiscale-res": "1 2 4 8", 
                "pipeline.model.spacetime-resolution": "64 64 64 25", 
            },
            {
                "pipeline.model.multiscale-res": "1 2 4 8", 
                "pipeline.model.spacetime-resolution": "64 64 64 50", 
            },
            {
                "pipeline.model.multiscale-res": "1 2 4 8", 
                "pipeline.model.spacetime-resolution": "64 64 64 100", 
            },
            {
                "pipeline.model.multiscale-res": "1 2 4 8 16", 
                "pipeline.model.spacetime-resolution": "64 64 64 25", 
            },
            {
                "pipeline.model.multiscale-res": "1 2 4 8 16", 
                "pipeline.model.spacetime-resolution": "64 64 64 50", 
            },
            {
                "pipeline.model.multiscale-res": "1 2 4 8 16", 
                "pipeline.model.spacetime-resolution": "64 64 64 100", 
            },
        ], None)
    exp.run(do_eval=False)

if __name__ == "__main__":
    main()
