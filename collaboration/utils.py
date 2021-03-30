import pandas as pd
import json
import numpy as np

def get_edits_pre_tag(data):
    ix = data['has_template'].idxmax()

    return data.loc[:ix]


def split_by_pagetitle():
    # can change logic if ordered!
    in_file_path='/srv/home/christinedk/wp_internship/data/talk_history/tmp_talk-weasel-meta-info.json'

    with open(in_file_path,'r') as in_json_file:
        for json_obj in in_json_file:
            json_obj_list = json.loads(json_obj)
            filename='talk_split/{}.json'.format(json_obj_list['page_id'])

            with open(filename, 'a+') as out_json_file:
                json.dump(json_obj, out_json_file, indent=4)


def read_revisions(filename, rename=False):
    revisions = pd.read_json(filename,lines=True)
    if rename:
        revisions = revisions.rename(columns = {'revision_timestamp':'event_timestamp','user_id':'event_user_id'})
    revisions['event_timestamp'] = pd.to_datetime(revisions['event_timestamp'])
    revisions = revisions.sort_values(by='event_timestamp', ascending=True)
    return revisions

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()