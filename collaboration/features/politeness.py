import numpy as np
from convokit import Corpus, User, Utterance
from convokit import TextParser
from convokit import PolitenessStrategies
from tqdm import tqdm
import datetime

def reconstruct_corpus(dataset):
    users = [utt['event_user_id'] for utt in dataset]
    users = np.unique(users)
    users_dict = {user: User(name=user) for user in users}

    utterances = []

    for utt in tqdm(dataset):
        user = users_dict[utt['event_user_id']] if utt['event_user_id'] is not None else users_dict['none']
        utterances.append(Utterance(id=utt['revision_id'], user=user, text=utt['event_comment\n']))

    corpus = Corpus(utterances=utterances)

    return corpus


def format_politeness_features(corpus):
    convs = list(corpus.iter_conversations())
    convs_feat_dict = {}

    for conv in tqdm(convs):
        utts = list(conv.iter_utterances())
        utts_feats = []

        for utt in utts:
            feat_dict = {}
            for feature, markers in utt.meta['politeness_markers'].items():
                feat_dict[feature] = len(markers)
            utts_feats.append(feat_dict)

        convs_feat_dict[conv.id] = utts_feats
    return convs_feat_dict


def get_utterance_details(utt):
    utt_dict = {}
    utt_dict['user'] = utt.speaker.id
    utt_dict['time'] = datetime.fromtimestamp(float(utt.timestamp))
    utt_dict['text'] = utt.text
    utt_dict['conv_id'] = utt.conversation_id
    utt_dict['id'] = utt.id
    utt_dict['reply_to'] = utt.reply_to
    return utt_dict


def get_politeness_features(dataset):
    corpus = reconstruct_corpus(dataset)

    print('creating corpus')
    parser = TextParser(verbosity=0)
    parsed_corpus = parser.transform(corpus)

    print('calculating politeness features')
    ps = PolitenessStrategies(verbose=False)
    awry_corpus = ps.transform(parsed_corpus, markers=True)

    print('formatting data')
    feat_dict = format_politeness_features(awry_corpus)

    politeness_dataset = []

    for d in dataset:
        conv_id = d[0]['conv_id']
        features = feat_dict[conv_id]
        politeness_dataset.append(features)

    return politeness_dataset


