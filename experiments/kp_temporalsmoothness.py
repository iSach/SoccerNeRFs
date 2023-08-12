"""
Tuning of temporal smoothness on Synthetic Balls.

Experiment. 5.3.3. Quantitative Results in Appendix A.2
"""
from ns_experiment import Experiment


def main():
    """
    Main fct.
    """
    exp = Experiment(
        name="temporalsmoothness",
        cam_path="data/stadium/camera_paths/nicecam.json",
        model="k-planes",
        dataset="stadium-data",
    )
    exp.set_params(
        [
            {
                "pipeline.model.loss-coefficients.time-smoothness-loss": 0.001,
                "pipeline.model.loss-coefficients.time-smoothness-proposal-loss": 0.001,
            },
            {
                "pipeline.model.loss-coefficients.time-smoothness-loss": 0.01,
                "pipeline.model.loss-coefficients.time-smoothness-proposal-loss": 0.01,
            },
            {
                "pipeline.model.loss-coefficients.time-smoothness-loss": 0.1,
                "pipeline.model.loss-coefficients.time-smoothness-proposal-loss": 0.1,
            },
            {
                "pipeline.model.loss-coefficients.time-smoothness-loss": 1.0,
                "pipeline.model.loss-coefficients.time-smoothness-proposal-loss": 1.0,
            },
        ], None)
    exp.run(do_eval=True)

if __name__ == "__main__":
    main()
