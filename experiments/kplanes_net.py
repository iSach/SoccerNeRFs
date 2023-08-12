"""
Experiments related to the decoder of K-Planes.

Experiment 5.5.3.
"""
from ns_experiment import Experiment


def main():
    """
    Main fct.
    """
    exp = Experiment(
        name="kplanes_net",
        cam_path="/workspace/data/synth_paderborn/camera_paths/around_player.json",
        model="k-planes",
        dataset="synthpaderborn-data",
    )
    exp.set_params(
        [
            {
                "pipeline.model.sigma-net-layers": "1",
                "pipeline.model.sigma-net-hidden-dim": 64,
                "pipeline.model.rgb-net-layers": 2,
                "pipeline.model.rgb-net-hidden-dim": 64,
                "pipeline.model.disable-viewing-dependent": True,
            },
            {
                "pipeline.model.sigma-net-layers": "1",
                "pipeline.model.sigma-net-hidden-dim": 64,
                "pipeline.model.rgb-net-layers": 2,
                "pipeline.model.rgb-net-hidden-dim": 64,
                "pipeline.model.disable-viewing-dependent": False,
            },
            {
                "pipeline.model.sigma-net-layers": "1",
                "pipeline.model.sigma-net-hidden-dim": 64,
                "pipeline.model.rgb-net-layers": 2,
                "pipeline.model.rgb-net-hidden-dim": 128,
                "pipeline.model.disable-viewing-dependent": True,
            },
            {
                "pipeline.model.sigma-net-layers": "1",
                "pipeline.model.sigma-net-hidden-dim": 128,
                "pipeline.model.rgb-net-layers": 2,
                "pipeline.model.rgb-net-hidden-dim": 64,
                "pipeline.model.disable-viewing-dependent": True,
            },
            {
                "pipeline.model.sigma-net-layers": "1",
                "pipeline.model.sigma-net-hidden-dim": 128,
                "pipeline.model.rgb-net-layers": 2,
                "pipeline.model.rgb-net-hidden-dim": 128,
                "pipeline.model.disable-viewing-dependent": True,
            },
            {
                "pipeline.model.sigma-net-layers": "1",
                "pipeline.model.sigma-net-hidden-dim": 128,
                "pipeline.model.rgb-net-layers": 2,
                "pipeline.model.rgb-net-hidden-dim": 128,
                "pipeline.model.disable-viewing-dependent": False,
            },
            {
                "pipeline.model.sigma-net-layers": "3",
                "pipeline.model.sigma-net-hidden-dim": 64,
                "pipeline.model.rgb-net-layers": 2,
                "pipeline.model.rgb-net-hidden-dim": 64,
                "pipeline.model.disable-viewing-dependent": True,
            },
            {
                "pipeline.model.sigma-net-layers": "1",
                "pipeline.model.sigma-net-hidden-dim": 64,
                "pipeline.model.rgb-net-layers": 3,
                "pipeline.model.rgb-net-hidden-dim": 64,
                "pipeline.model.disable-viewing-dependent": True,
            },
            {
                "pipeline.model.sigma-net-layers": "3",
                "pipeline.model.sigma-net-hidden-dim": 64,
                "pipeline.model.rgb-net-layers": 3,
                "pipeline.model.rgb-net-hidden-dim": 64,
                "pipeline.model.disable-viewing-dependent": True,
            },
            {
                "pipeline.model.sigma-net-layers": "3",
                "pipeline.model.sigma-net-hidden-dim": 64,
                "pipeline.model.rgb-net-layers": 3,
                "pipeline.model.rgb-net-hidden-dim": 64,
                "pipeline.model.disable-viewing-dependent": False,
            },
            {
                "pipeline.model.sigma-net-layers": "1",
                "pipeline.model.sigma-net-hidden-dim": 64,
                "pipeline.model.rgb-net-layers": 1,
                "pipeline.model.rgb-net-hidden-dim": 64,
                "pipeline.model.disable-viewing-dependent": True,
            },
        ], None)
    exp.run(do_eval=False)

if __name__ == "__main__":
    main()
