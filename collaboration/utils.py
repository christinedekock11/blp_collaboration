import pandas as pd

def get_edits_pre_tag(data):
    ix = data['has_template'].idxmax()

    return data.loc[:ix]
