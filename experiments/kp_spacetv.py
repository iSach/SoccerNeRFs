"""
Tuning of Space TV on Synthetic Balls.

Experiment. 5.3.3. Quantitative Results in Appendix A.2
"""
from ns_experiment import Experiment


def main():
    """
    Main fct.
    """
    exp = Experiment(
        name="kp_spacetv",
        cam_path="data/stadium/camera_paths/nicecam.json",
        model="k-planes",
        dataset="stadium-data",
    )
    exp.set_params(
        [
            {
                "pipeline.model.loss-coefficients.space-tv-loss": 0.0002,
                "pipeline.model.loss-coefficients.space-tv-proposal-loss": 0.0002,
            },
            {
                "pipeline.model.loss-coefficients.space-tv-loss": 0.002,
                "pipeline.model.loss-coefficients.space-tv-proposal-loss": 0.002,
            },
            {
                "pipeline.model.loss-coefficients.space-tv-loss": 0.02,
                "pipeline.model.loss-coefficients.space-tv-proposal-loss": 0.02,
            },
            {
                "pipeline.model.loss-coefficients.space-tv-loss": 2.0,
                "pipeline.model.loss-coefficients.space-tv-proposal-loss": 2.0,
            },
        ], None)
    exp.run(do_eval=True)

if __name__ == "__main__":
    main()