import spikeinterface as si
from typing import Tuple
from spikeinterface.curation import MergeUnitsSorting, get_potential_auto_merge

def main():
    """
    :rtype: object
    """
    print("main")

def get_potential_merge(sorting, wf):
    """
    :param sorting:
    :param wf:
    :rtype: Tuple[si.BaseSorting, si.WaveformExtractor]
    """
    print("get_potential_merge")
    merges = get_potential_auto_merge(wf, minimum_spikes=1000,  maximum_distance_um=150.,
                                  peak_sign="neg", bin_ms=0.25, window_ms=100.,
                                  corr_diff_thresh=0.16, template_diff_thresh=0.25,
                                  censored_period_ms=0., refractory_period_ms=1.0,
                                  contamination_threshold=0.2, num_channels=5, num_shift=5,
                                  firing_contamination_balance=1.5)
    

    if merges:  # This will be False if units_to_merge is an empty list
        clean_sorting = MergeUnitsSorting(parent_sorting=sorting, units_to_merge=merges, properties_policy='keep', delta_time_ms=0.4)
    else:
        clean_sorting=sorting
    # handle the case when there are no units to merge
    return clean_sorting

if __name__ == "__main__":
    main()
    get_potential_merge(si.BaseSorting, si.WaveformExtractor)
