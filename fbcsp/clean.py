from collections import namedtuple
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne

CleanResult = namedtuple('CleanResult', ['rejected_chan_names',
    'rejected_trials', 'clean_trials', 'rejected_max_min',
    'rejected_var'])


class NoCleaner():
    def __init__(self, epoch_ival_ms, name_to_start_codes):
        self.name_to_start_codes = name_to_start_codes
        self.epoch_ival_ms = epoch_ival_ms

    def clean(self, cnt, ignore_chans=False):
        # Segment into trials and take all! :)
        # Segment just to select markers and kick out out of bounds
        # trials
        # chans ignored always anyways... so ignore_chans parameter does not
        # matter
        epo = create_signal_target_from_raw_mne(cnt,
            name_to_start_codes=self.name_to_start_codes,
                       epoch_ival_ms=self.epoch_ival_ms)
        clean_trials = list(range(epo.X.shape[0]))

        clean_result = CleanResult(rejected_chan_names=[],
                                   rejected_trials=[],
                                   clean_trials=clean_trials,
                                   rejected_max_min=[],
                                   rejected_var=[])
        return clean_result