"""
Studies the depth loss coefficient impact on K-Planes (Paderborn).

Not in report.
"""
from ns_experiment import Experiment


def main():
    """
    Main fct.
    """
    exp = Experiment(
        name="depth_k-planes_mask",
        cam_path="data/paderborn/camera_paths/bounded_smallbox540.json",
        model="k-planes",
        dataset="paderborn-data",
    )
    exp.set_params(
        [
            {
                "pipeline.model.loss-coefficients.depth-loss": 0.001,
            },
            {
                "pipeline.model.loss-coefficients.depth-loss": 0.01,
            },
            {
                "pipeline.model.loss-coefficients.depth-loss": 0.05,
            },
            {
                "pipeline.model.loss-coefficients.depth-loss": 0.1,
            },
            {
                "pipeline.model.loss-coefficients.depth-loss": 0.5,
            },
            {
                "pipeline.model.loss-coefficients.depth-loss": 1.0,
            },
            {
                "pipeline.model.loss-coefficients.depth-loss": 10.0,
            },
        ], None)
    exp.run(do_eval=False)

if __name__ == "__main__":
    main()
