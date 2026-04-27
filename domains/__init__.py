from .ppt import run_ppt_analysis
from .hilton import run_hilton_analysis
from .netflix import run_netflix_analysis
from .spotify import run_spotify_analysis
from .godaddy import run_godaddy_analysis


def run_analysis(df, domain, id_col, text_col, validation_dict,
                 progress_cb=None, rule_threshold=0.7):
    if domain == "hilton":
        return run_hilton_analysis(df, id_col, text_col, validation_dict, progress_cb)
    elif domain == "netflix":
        return run_netflix_analysis(df, id_col, text_col, validation_dict, progress_cb, rule_threshold)
    elif domain == "spotify":
        return run_spotify_analysis(df, id_col, text_col, validation_dict, progress_cb)
    elif domain == "godaddy":
        return run_godaddy_analysis(df, id_col, text_col, validation_dict, progress_cb, rule_threshold)
    else:
        return run_ppt_analysis(df, id_col, text_col, validation_dict, progress_cb, rule_threshold)
