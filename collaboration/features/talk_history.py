import re
from parser import ParserError

from bs4 import BeautifulSoup
from dateutil import parser as dateparser
import string

from convokit import Corpus, Speaker, Utterance, TextParser, PolitenessStrategies
import numpy as np
import pandas as pd


class ConvParser(object):

    def __init__(self):
        self.date_regex = re.compile('(\d\d:\d\d,\s\d+\s[A-z]*\s\d{4})')
        self.utt_regex = re.compile('(\[\[User:.+\]\])(.*\(UTC\))?(?:$|\n)')
        self.user_regex = re.compile(r'\[\[(User:.*?)\|')
        self.section_regex = re.compile('(==+.*==+)\s*')

    def get_sections(self, text):
        secs = self.section_regex.split(text)
        tups = []

        for i in np.arange(len(secs)):
            if secs[i].startswith('=='):
                tups.append((secs[i], secs[i + 1]))
            else:
                if secs[i - 1].startswith('=='):
                    pass
                else:
                    tups.append((None, secs[i]))
        return tups

    def get_conversation_attributes(self,conv, page_id):
        conv = pd.DataFrame(conv)
        conv['text'] = conv.text.str.lstrip()
        conv['user'] = conv.user.apply(self.format_username).fillna('unknown')
        conv['date'] = conv.date.apply(self.format_date).fillna('unknown')
        conv['indent_depth'] = conv.text.apply(lambda x: (len(x) - len(x.lstrip(':'))))
        conv['conv_id'] = str(page_id) + conv.sec_num.astype(str)
        conv['utt_id'] = conv.conv_id + '.' + conv.utt_ind.astype(str)
        return conv.to_dict('records')

    def get_conversation_structure(self,text):
        secs = self.get_sections(text)
        convs = []
        for secnum, sections in enumerate(secs):
            secname, sec = sections
            utts = self.utt_regex.split(sec)
            if len(utts) == 1:
                conv = [{'text': utts[0], 'user': 'unknown', 'date': 'unknown',
                         'sec_num': secnum, 'sec_name': secname, 'utt_ind':0}]
            else:
                conv = [{'text': u[0], 'user': u[1], 'date': u[2],
                         'sec_num': secnum, 'sec_name': secname, 'utt_ind':i}
                            for i,u in enumerate(zip(*[iter(utts)] * 3))]
            convs.append(conv)
        return convs

    def format_conv(self, text, page_id):
        soup = BeautifulSoup(text, features="lxml")
        text = soup.get_text()

        conversations = self.get_conversation_structure(text)
        conversations = np.concatenate([self.get_conversation_attributes(conv, page_id) for conv in conversations])

        return conversations

    def format_username(self, name):
        match = self.user_regex.search(name)
        if match:
            return match.group(1)
        else:
            return name

    def format_date(self, date=''):
        match = self.date_regex.search(date) if date else None
        date = match.group(1) if match else None
        #        date = dateparser.parse(match.group(1))
        #    except ParserError:
        #        date = None
        return date

class FeatureExtractor(object):
    def __init__(self):
        self.text_parser = TextParser(verbosity=0)
        self.politeness_parser = PolitenessStrategies(verbose=False)

    def format_as_corpus(self, conv):
        users = np.unique([utt['user'] for utt in conv])
        users_dict = {user: Speaker(name=user) for user in users}

        utterances = []

        for utt in conv:
            user = users_dict[utt['user']]
            utt_obj = Utterance(id=utt['utt_id'], user=user,
                                text=utt['text'], root=str(utt['conv_id']))
            utt_obj.add_meta('reply_depth', utt['indent_depth'])
            utterances.append(utt_obj)

        corpus = Corpus(utterances=utterances)

        return corpus

    @staticmethod
    def format_politeness_features(corpus):
        convs_feat_dict = {}

        for conv in corpus.iter_conversations():
            utts_feats = []

            for utt in conv.iter_utterances():
                feat_dict = {}
                for feature, markers in utt.meta['politeness_markers'].items():
                    feat_dict[feature] = len(markers)
                feat_dict['reply_depth'] = utt.meta['reply_depth']
                utts_feats.append(feat_dict)

            convs_feat_dict[conv.id] = utts_feats
        return convs_feat_dict

    def get_language_features(self,conv):
        conv = self.format_as_corpus(conv)
        parsed_conv = self.text_parser.transform(conv)
        awry_conv = self.politeness_parser.transform(parsed_conv, markers=True)
        feat_dict = self.format_politeness_features(awry_conv)
        return feat_dict

