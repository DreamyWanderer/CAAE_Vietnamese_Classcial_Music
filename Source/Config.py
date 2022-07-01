from bidict import bidict

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