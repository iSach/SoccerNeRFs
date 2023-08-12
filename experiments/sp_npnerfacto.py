"""
Use of NeRFPlayer-Nerfacto on SynthPaderborn dataset.

Experiment 5.5.1.
"""
from ns_experiment import Experiment


def main():
    """
    Main fct.
    """
    exp = Experiment(
        name="sp_npnerfacto",
        cam_path="/workspace/data/synth_paderborn/camera_paths/around_player.json",
        model="nerfplayer-nerfacto",
        dataset="synthpaderborn-data",
    )
    exp.set_params(
        [
            {
                "pipeline.model.log2-hashmap-size": 18,
            },
            {
                "pipeline.model.log2-hashmap-size": 19,
            },
        ], None)
    exp.run(do_eval=True)

if __name__ == "__main__":
    main()
