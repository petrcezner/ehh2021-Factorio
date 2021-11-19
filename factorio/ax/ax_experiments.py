from pathlib import Path
from typing import Iterable, List, Tuple
from ax.service.ax_client import AxClient

from factorio.ax.gp_problem import GpProblem


class ExperimentHolder(AxClient):
    """Object that gathers eperiments settings and results and reasons
    about next optimal settings to try.
    Once the object is created, call functions:
    exp_holder = ExperimentHolder(params_specification)
    parameters, trial_index = exp_holder.get_next_trial()
    exp_holder.complete_trial(trial_index=trial_index, raw_data=results)
    exp_holder.log_trial_failure(trial_index=trial_index)
    """

    def __init__(self,
                 parameters: List[dict],
                 exp_name: str = None,
                 objective_name: str = 'score',
                 problem: GpProblem = None,
                 tracking_metric_names: List[str] = None,
                 minimize=False):
        super().__init__()
        tracking_metric_names = tracking_metric_names if tracking_metric_names is not None else [objective_name]
        self.create_experiment(
            name=exp_name,
            parameters=parameters,
            objective_name=objective_name,
            tracking_metric_names=tracking_metric_names,
            minimize=minimize,
        )
        self.current_trial_index = None
        self.current_pars = None
        self.problem = problem
        self._can_move_on = True

    def add_historic_trials_results(self, trials: Iterable[Tuple[dict, float]]):
        for trial in trials:
            pars, trial_idx = self.attach_trial(parameters=trial[0])
            super().complete_trial(trial_index=trial_idx, raw_data=trial[1])

    def get_next_trial(self):
        if self._can_move_on:
            pars, trial_index = super().get_next_trial()
            self.current_trial_index = trial_index
            self.current_pars = pars
            self._can_move_on = False
        return self.current_pars

    def complete_trial(self):
        if not self._can_move_on:
            # TODO: call problem function
            score = 1
            score_dict = {'score': score}
            super().complete_trial(trial_index=self.current_trial_index,
                                   raw_data=score_dict)
            self._can_move_on = True

    def trial_failed(self):
        if not self._can_move_on:
            super().log_trial_failure(trial_index=self.current_trial_index)
            self._can_move_on = True


if __name__ == '__main__':
    import numpy as np
    import datetime
    from tqdm import trange

    experiment_name = 'Chleba'
    N_trials = 10

    time_now = datetime.datetime.utcnow()
    experiment_root = Path(f'.out/experiments/{time_now.strftime("%Y-%m-%d")}')
    experiment_root.mkdir(parents=True, exist_ok=True)

    spectral_ker_names = []
    for i in range(1, 4):
        spectral_ker_names.append(f'spectral_{i}')

    poly_ker_names = []
    for i in range(1, 4):
        poly_ker_names.append(f'poly_{i}')

    spectral_ker_names.append('None')
    poly_ker_names.append('None')

    params = [
        {
            'name': "ker1",
            'value_type': 'str',
            'type': 'choice',
            'values': ['rbf', 'matern15', 'matern05', 'None'],
        },
        {
            'name': "ker3",
            'value_type': 'str',
            'type': 'choice',
            'values': spectral_ker_names,
        }
    ]

    exp_service_object = ExperimentHolder(params, exp_name=experiment_name)

    file_name = Path(f'{experiment_name}_{time_now.strftime("%Y%d%m%H%M%S")}.json')
    full_path = str(experiment_root / file_name)
    for i in trange(N_trials):
        parameters = exp_service_object.get_next_trial()
        if i == 8:
            exp_service_object.trial_failed()
            continue
        exp_service_object.complete_trial(np.random.randn())
        exp_service_object.save_to_json_file(full_path)

    best_parameters, values = exp_service_object.get_best_parameters()

    print(f'best parameters: {best_parameters} with expected score: {values[0]}')
    print(f'Done')
