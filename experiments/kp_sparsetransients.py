"""
Tuning of Sparse Transients on Synthetic Balls.

Experiment. 5.3.3. Quantitative Results in Appendix A.2
"""
from ns_experiment import Experiment


def main():
    """
    Main fct.
    """
    exp = Experiment(
        name="kp_sparsetransients",
        cam_path="data/stadium/camera_paths/nicecam.json",
        model="k-planes",
        dataset="stadium-data",
    )
    exp.set_params(
        [
            {
                "pipeline.model.loss-coefficients.sparse-transients-loss": 0.0001,
                "pipeline.model.loss-coefficients.sparse-transients-proposal-loss": 0.0001,
            },
            {
                "pipeline.model.loss-coefficients.sparse-transients-loss": 0.001,
                "pipeline.model.loss-coefficients.sparse-transients-proposal-loss": 0.001,
            },
            {
                "pipeline.model.loss-coefficients.sparse-transients-loss": 0.1,
                "pipeline.model.loss-coefficients.sparse-transients-proposal-loss": 0.1,
            },
            {
                "pipeline.model.loss-coefficients.sparse-transients-loss": 10.0,
                "pipeline.model.loss-coefficients.sparse-transients-proposal-loss": 10.0,
            },
        ], None)
    exp.run(do_eval=True)

if __name__ == "__main__":
    main()