"""
Experiments were done on HyperNeRF to debug the implementation of K-Planes (and compared with other SOTA models).

Not in report.
"""
from ns_experiment import Experiment


def main():
    """
    Main fct.
    """
    exp = Experiment(
        name="hypernerf",
        cam_path=None,
        model="k-planes",
        dataset="hypernerf-data",
    )
    exp.set_params(
        [
            {
                "pipeline.model.bounded": False,
                "data": "data/hypernerf/3dprinter",
            },
            {
                "pipeline.model.bounded": False,
                "data": "data/hypernerf/broom",
            },
            {
                "pipeline.model.bounded": False,
                "data": "data/hypernerf/chicken",
            },
            {
                "pipeline.model.bounded": False,
                "data": "data/hypernerf/banana",
            },
        ], None)
    exp.run(do_eval=True)

if __name__ == "__main__":
    main()
