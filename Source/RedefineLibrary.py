from operator import attrgetter
from typing import TYPE_CHECKING, Union, List
import muspy

import numpy as np
from numpy import ndarray
from pypianoroll import Multitrack, Track
from pypianoroll import Track as PypianorollTrack
import pypianoroll

from muspy import DEFAULT_VELOCITY, Note, DEFAULT_RESOLUTION, Music, Tempo, Track

if TYPE_CHECKING:
    from muspy import Music

def to_pianoroll_representation_holding(
    music: "Music",
    encode_velocity: bool = True,
    dtype: Union[np.dtype, type, str] = None,
) -> ndarray:

    if dtype is None:
        dtype = np.int16 if encode_velocity else bool

    # Collect notes
    notes = []
    for track in music.tracks:
        notes.extend(track.notes)

    # Raise an error if no notes are found
    if not notes:
        raise RuntimeError("No notes found.")

    # Sort the notes
    notes.sort(key=attrgetter("time", "pitch", "duration", "velocity"))

    if not notes:
        return np.zeros((0, 128), dtype)

    # Initialize the array
    length = max((note.end for note in notes))
    array = np.zeros((length + 1, 128), dtype)

    # Encode notes
    for note in notes:
        if note.velocity is not None:
            if encode_velocity:
                array[note.time : note.end, note.pitch] = note.velocity
            else:
                array[note.time : note.end, note.pitch] = note.velocity > 0
        elif encode_velocity:
            array[note.time : note.end, note.pitch] = DEFAULT_VELOCITY
        else:
            array[note.time : note.end, note.pitch] = True

        #Changed part. If the previous timestep is played, this note is a seperate note: index by negative
        #Use try-except for 0 timestep case

        try:
            if array[note.time - 1, note.pitch] != 0:
                array[note.time, note.pitch] = -array[note.time, note.pitch]
        except:
            pass

    return array

def findRangeSeperateNote(start: int, limit: int, pitch: int, array: ndarray):

    i = start + 1
    end = limit

    while i < limit:
        if array[i, pitch] < 0:
            end = i
            break
        else:
            i += 1 

    return (start, end)

def _pianoroll_to_notes(
    array: ndarray, encode_velocity: bool, default_velocity: int
) -> List[Note]:
    binarized = abs(array) > 0
    diff = np.diff(binarized, axis=0, prepend=0, append=0)
    notes = []
    for i in range(128):
        boundaries = np.nonzero(diff[:, i])[0]
        for note_idx in range(len(boundaries) // 2):
            start = boundaries[2 * note_idx]
            end = boundaries[2 * note_idx + 1]
            if encode_velocity:
                velocity = array[start, i]
            else:
                velocity = default_velocity

            s = int(start)
            e = None
            while s < int(end):
                s, e = findRangeSeperateNote(s, end, i, array)

                note = Note(
                    time=s,
                    pitch=i,
                    duration= e - s,
                    velocity=abs(int(array[s, i])),
                )
                notes.append(note)

                s = e

    notes.sort(key=attrgetter("time", "pitch", "duration", "velocity"))

    return notes
    
def from_pianoroll_representation_holding(
    array: ndarray,
    resolution: int = DEFAULT_RESOLUTION,
    program: int = 0,
    is_drum: bool = False,
    encode_velocity: bool = True,
    default_velocity: int = DEFAULT_VELOCITY,
) -> Music:

    if encode_velocity and not np.issubdtype(array.dtype, np.integer):
        raise TypeError(
            "Array must be of type int when `encode_velocity` is True."
        )
    if not encode_velocity and not np.issubdtype(array.dtype, np.bool):
        raise TypeError(
            "Array must be of type bool when `encode_velocity` is False."
        )

    # Convert piano roll to notes
    notes = _pianoroll_to_notes(array, encode_velocity, default_velocity)

    # Create the Track and Music objects
    track = Track(program=program, is_drum=is_drum, notes=notes)
    music = Music(resolution=resolution, tracks=[track])

    return music