from bidict import bidict

import os

emotion = bidict(
    {
        'elated': (1, 1),
        'serene': (1, -1),
        'sad': (-1, -1),
        'tense': (-1, 1)
    }
)

majorkey = bidict(
    {
        'C major': (0, 0),
        'C# major': (1, 0),
        'D- major': (1, 1),
        'D major': (2, 0),
        'D# major': (3, 0),
        'E- major': (3, 1),
        'E major': (4, 0),
        'F- major': (4, 1),
        'F major': (5, 0),
        'F# major': (6, 0),
        'G- major': (6, 1),
        'G major': (7, 0),
        'G# major': (8, 0),
        'A- major': (8, 1),
        'A major': (9, 0),
        'A# major': (10, 0),
        'B- major': (10, 1),
        'B major': (11, 0),
        'C- major': (11, 1)
    }
)

minorkey = bidict(
    {
        'C minor': (0, 0),
        'C# minor': (1, 0),
        'D- minor': (1, 1),
        'D minor': (2, 0),
        'D# minor': (3, 0),
        'E- minor': (3, 1),
        'E minor': (4, 0),
        'F- minor': (4, 1),
        'F minor': (5, 0),
        'F# minor': (6, 0),
        'G- minor': (6, 1),
        'G minor': (7, 0),
        'G# minor': (8, 0),
        'A- minor': (8, 1),
        'A minor': (9, 0),
        'A# minor': (10, 0),
        'B- minor': (10, 1),
        'B minor': (11, 0),
        'C- minor': (11, 1)
    }
)

keyEmotion = {
    'C major': 'serene',
    'C# major': 'sad',
    'D major': 'elated',
    'D# major': 'tense',
    'E major': 'tense',
    'F major': 'elated',
    'F# major': 'tense',
    'G major': 'serene',
    'G# major': 'sad',
    'A major': 'serene',
    'A# major': 'elated',
    'B major': 'tense',
    'C minor': 'tense',
    'C# minor': 'sad',
    'D minor': 'sad',
    'D# minor': 'sad',
    'E minor': 'tense',
    'F minor': 'sad',
    'F# minor': 'serene',
    'G minor': 'tense',
    'G# minor': 'tense',
    'A minor': 'serene',
    'A# minor': 'sad',
    'B minor': 'serene',
}

big_cag = {"classical": 1, "vietnam": 2}
small_cag = [
    {
        "elated": 0,
        "serene": 1,
        "sad": 2,
        "tense": 3
    },
    {
        "elated": 4,
        "serene": 5,
        "sad": 6,
        "tense": 7
    }
]

coding_size = 512
num_type = 8
batch_size = 32
num_epoch = 100
num_feature = 130

root_logdir = os.path.join(os.curdir, "my_logs")

encoder_param = {
    "depth": 3,
    "num_hidden_node": 1024
}

decoder_param = {
    "depth": 3,
    "num_hidden_node": 1024
}

encoder_dis = {
    "depth": 2,
    "num_hidden_node": 512
}

decoder_dis = {
    "depth": 2,
    "num_hidden_node": 1024
}