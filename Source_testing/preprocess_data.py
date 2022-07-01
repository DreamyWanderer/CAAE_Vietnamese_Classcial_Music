from turtle import screensize
import muspy
import time
import matplotlib.pyplot as plt
import pretty_midi
import music21

midi_data = pretty_midi.PrettyMIDI('Dataset\\beeth\\mond_3.mid')
print( midi_data.resolution )
score = muspy.from_pretty_midi(midi_data, resolution = 24)

i = 1
while i < len(score.tracks):
    tmp = score.tracks[i].notes
    score.tracks[0].notes.extend(tmp)
    score.tracks.pop(i)

print(score.tempos)

muspy.write_midi('Dataset\\beeth\\mond_3_test.mid', score, "pretty_midi")

piece = music21.converter.parse('Dataset\\beeth\\mond_3_test.mid')
key = piece.analyze('key')
print(key)
