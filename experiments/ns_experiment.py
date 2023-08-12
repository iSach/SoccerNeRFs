"""
Util class for Nerfstudio experiments

Check any experiment's .py file for example usage :-)
"""

import os
from typing import List, Dict
import json

class Experiment:
    """
    Config for one experiment.
    """
    name: str
    cam_path: str
    model: str
    dataset: str
    params: List[Dict] = None
    data_params: List[Dict] = None
    _active_params: Dict = None
    _active_data_params: Dict = None

    def __init__(self, name, cam_path, model, dataset=""):
        self.name = name
        self.cam_path = cam_path
        self.model = model
        self.dataset = dataset
        self.params = None
        self.data_params = None
        self._active_params = None
        self._active_data_params = None

        # mkdir
        os.system(f"mkdir -p {self.get_path()}")
        os.system(f"mkdir -p {self.get_path()}/renders")
        os.system(f"mkdir -p {self.get_path()}/evals")

    def run(self, do_eval=False):
        """
        Run the experiment with all runs.
        """
        assert self.params is not None, "Params not set"

        for i, param in enumerate(self.params):
            self._active_params = param
            if self.data_params is not None:
                self._active_data_params = self.data_params[i]
            self.__train()
            self.__render()
            if do_eval:
                self.__eval()

    def set_params(self, params, data_params=None):
        """
        Set parameters lists for the experiment.
        """
        self.params = []
        for param in params:
            self.params.append({k: str(v) for k, v in param.items()})
        if data_params is not None:
            if isinstance(data_params, dict):
                data_params = [data_params] * len(params)
            else:
                self.data_params = []
                for param in data_params:
                    self.data_params.append({k: str(v) for k, v in param.items()})

        # Dump params as json
        with open(self.get_path() + "/params.json", "w") as f:
            json.dump(self.params, f, indent=4)

    def __train(self):
        """
        Train the current run.
        """
        assert self._active_params is not None, "Active params not set"
        params = self._active_params

        cmd = f"ns-train {self.model}"
        for cmd_param, param in params.items():
            cmd += f" --{cmd_param} {param}"
        cmd += f" --experiment-name {self.name}"
        cmd += f" --timestamp {self.__get_timestamp()}"
        cmd += f" {self.dataset}"
        if self._active_data_params is not None:
            for cmd_param, param in self._active_data_params.items():
                cmd += f" --{cmd_param} {param}"
        print("Running: ", cmd)
        os.system(cmd)

    def __render(self):
        """
        Render the current run.
        """
        cam_path = self.cam_path
        if not isinstance(cam_path, list):
            cam_path = [cam_path]
        for i, path in enumerate(cam_path):
            cmd = "ns-render" + \
                f" --load-config {self.get_config_path()}" + \
                " --traj filename" + \
                " --eval-num-rays-per-chunk 65536" + \
                f" --camera-path-filename {path}" + \
                f" --output-path {self.get_render_path(i)}"
            print("Running: ", cmd)
            os.system(cmd)

    def __eval(self):
        """
        Runs evaluation on the current run.
        """
        cmd = "ns-eval" + \
            f" --load-config {self.get_config_path()}" + \
            f" --output-path {self.get_eval_path()}"
        print("Running: ", cmd)
        os.system(cmd)

    def get_path(self):
        """
        Returns the path of the experiment.
        """
        return f"/workspace/outputs/{self.name}"

    def get_run_path(self):
        """
        Returns the path to the current run.
        """
        return self.get_path() + f"/{self.model}/{self.__get_timestamp()}"
   
    def get_config_path(self):
        """
        Returns the path to the config file for the current run.
        """
        return self.get_run_path() + "/config.yml"

    def get_render_path(self, cam_path_id=0):
        """
        Returns the path to the render file for the current run.
        """
        if isinstance(self.cam_path, list):
            cam_path_name = self.cam_path[cam_path_id].split("/")[-1].split(".")[0]
            return self.get_path() + f"/renders/{self.__get_timestamp()}/{cam_path_name}.mp4"
        return self.get_path() + f"/renders/{self.__get_timestamp()}.mp4"

    def get_eval_path(self):
        """
        Returns the path to the eval file for the current run.
        """
        return self.get_path() + f"/evals/{self.__get_timestamp()}.json"

    def __get_timestamp(self):
        """
        Returns a formatted timestamp for the current run.
        """
        assert self._active_params is not None, "Active params not set"

        timestamp = ""
        param_values = list(self._active_params.values())
        if self._active_data_params is not None:
            param_values = param_values + list(self._active_data_params.values())
        for val in param_values:
            # 0.01 -> 0_01
            timestamp += str(val).replace(".", "_").replace(" ", "_") + "-"
        return timestamp[:-1]