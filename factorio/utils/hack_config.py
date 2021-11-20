import configparser
from pathlib import Path


class HackConfig:
    def __init__(self, z_case, data_frequency, trials, inter_trials, experiment_name, valid_size, use_gpu,
                 cv_ratios_start, cv_ratios_stop, cv_ratios_steps, teams, data_folder, hospital):
        self.z_case = z_case
        self.data_frequency = data_frequency
        self.trials = trials
        self.inter_trials = inter_trials
        self.experiment_name = experiment_name
        self.valid_size = valid_size
        self.use_gpu = use_gpu
        self.cv_ratios_start = cv_ratios_start
        self.cv_ratios_stop = cv_ratios_stop
        self.cv_ratios_steps = cv_ratios_steps
        self.teams = teams
        self.data_folder = data_folder
        self.hospital = hospital

    @classmethod
    def from_config(cls, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)

        z_case = Path(config['HOSPITAL'].get('data_path'))
        data_folder = Path(config['HOSPITAL'].get('data_folder'))
        data_frequency = config['HOSPITAL'].getint('data_frequency', fallback=60)
        hospital = config['HOSPITAL'].get('hospital')

        trials = config['ax'].getint('trials', fallback=10)
        inter_trials = config['ax'].getint('inter_trials', fallback=10)
        valid_size = config['ax'].getint('valid_size', fallback=10)
        experiment_name = config['ax'].get('experiment_name', fallback='test')
        use_gpu = config['ax'].getboolean('use_gpu', fallback=False)
        cv_ratios_start = config['ax'].getfloat('cv_ratios_start', fallback=0.4)
        cv_ratios_stop = config['ax'].getfloat('cv_ratios_stop', fallback=0.9)
        cv_ratios_steps = config['ax'].getint('cv_ratios_steps', fallback=100)

        i = 0
        teams = {'teams': []}
        while f'{i}.name' in config['football']:
            teams['teams'].append({'name': config['football'].get(f'{i}.name'),
                                   'capacity': config['football'].getint(f'{i}.capacity')})
            i += 1

        return cls(z_case=z_case,
                   data_frequency=data_frequency,
                   teams=teams,
                   trials=trials,
                   inter_trials=inter_trials,
                   valid_size=valid_size,
                   experiment_name=experiment_name,
                   use_gpu=use_gpu,
                   cv_ratios_start=cv_ratios_start,
                   cv_ratios_stop=cv_ratios_stop,
                   cv_ratios_steps=cv_ratios_steps,
                   data_folder=data_folder,
                   hospital=hospital)
