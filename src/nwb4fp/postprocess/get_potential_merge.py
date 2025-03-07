import spikeinterface as si
from typing import Tuple
from spikeinterface.curation import MergeUnitsSorting, get_potential_auto_merge,remove_duplicated_spikes,remove_excess_spikes,remove_redundant_units

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
    global_job_kwargs = dict(n_jobs=12, total_memory="64G",mp_context= "spawn",progress_bar=True, chunk_size=5000, chunk_duration="1s")
    si.set_global_job_kwargs(**global_job_kwargs)
    merges = get_potential_auto_merge(wf, min_spikes=1000,  max_distance_um=150.)
    

    if merges:  # This will be False if units_to_merge is an empty list
        clean_sorting = MergeUnitsSorting(sorting, units_to_merge=merges, properties_policy='keep', delta_time_ms=0.4)
    else:
        clean_sorting=sorting
    # handle the case when there are no units to merge
    #clean_sorting = remove_redundant_units(clean_sorting)
   
    return remove_duplicated_spikes(clean_sorting)

if __name__ == "__main__":
    main()
    get_potential_merge(si.BaseSorting, si.WaveformExtractor)
