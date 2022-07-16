import muspy
import pretty_midi
import music21
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import Config

from RedefineLibrary import to_pianoroll_representation_holding
from RedefineLibrary import from_pianoroll_representation_holding

muspy.to_pianoroll_representation = to_pianoroll_representation_holding
muspy.from_pianoroll_representation = from_pianoroll_representation_holding 

id_pianoroll = 1
id_sample = 0

def createMetadaOriginalDataset(midiPath):
    '''
    Tạo các metadata kiểu json mô tả các tác phẩm trong thư mục Dataset_original. Hàm mang tính đặc thù và chỉ dùng trong cấu trúc file như mô tả trong Data_process.md.

    Input -- đường dẫn tới file nhạc midi

    Output -- tạo một file json có tên trùng tên file nhạc với các mô tả
    '''

    midiPath = os.path.splitext(midiPath)[0]
    listPathComponent = os.path.normpath(midiPath).split(os.path.sep)
    
    if listPathComponent[-2] == 'Vietnamese_dataset':
        style = "vietnam"
    else:
        style = "classical" 

    metadata = {
        "name": os.path.basename(midiPath),
        "style": style,
        "emotion": None,
        "key": None
    }
    json_object = json.dumps(metadata, indent = 4)
    with open("Dataset\Dataset_original\Metadata\\" + metadata["name"] + ".json", "w") as f:
        f.write(json_object)

def createPianoRepresentation(score: muspy.Music):

    listFirstBeat = inferBeat(score)
    listTempo = inferTempo(score)
    score_pianoroll = muspy.to_pianoroll_representation(score)
    score_pianoroll = np.column_stack((score_pianoroll, listFirstBeat))
    score_pianoroll = np.column_stack((score_pianoroll, listTempo))

    return score_pianoroll

def normalizedDataset(midiPath):
    '''
    Chuẩn hóa các bản nhạc trong Dataset_original về dạng biểu diễn pianoroll chuẩn bị trực tiếp cho việc tạo các sample, chứa trong thư mục Dataset_normalized.

    Input -- Đường dẫn tới file nhạc midi cần chuẩn hóa
            id: số thứ tự đánh dấu pianoroll trong thư mục Dataset_normalized

    Output -- tạo một file npz chứa dạng biểu diễn pianoroll của bản nhạc sau khi đã chuẩn hóa 
    '''

    global id_pianoroll

    JSONpath = "Dataset\Dataset_original\Metadata\\" + os.path.basename(os.path.splitext(midiPath)[0]) + ".json"
    JSONpath_normalized = "Dataset\Dataset_normalized\\" + str(id_pianoroll) + ".json"

    midi_data = pretty_midi.PrettyMIDI(midiPath)
    score = muspy.from_pretty_midi(midi_data, 24) # A quaternote has 24 time step

    groupTracks(score)
    score.infer_barlines_and_beats()

    key = inferKey(score)
    score_pianoroll = createPianoRepresentation(score)
    
    np.save("Dataset\Dataset_normalized\\" + str(id_pianoroll) + ".npy", score_pianoroll)

    #Lưu thông tin key của bản nhạc vào metadata gốc đã tạo
    dic = openJSONFile(JSONpath)
    dic["key"] = str(key.name)
    writeJSONFile(JSONpath, dic)
    
    #Tạo thông tin của pianoroll đã chuẩn hóa, cập nhật từ metadate của bản nhạc tại thư mục Dataset_original
    dic_normalized = createMetadataNormalized(JSONpath_normalized)
    dic_normalized.update(dic)
    dic_normalized["id"] = id_pianoroll
    writeJSONFile(JSONpath_normalized, dic_normalized)
    id_pianoroll += 1

    augmentateTempo(score_pianoroll, 130, dic_normalized)
    augmentateTempo(score_pianoroll, 70, dic_normalized)
    augmentateKey(score, dic, key)

def groupTracks(score):
    '''
    Nếu Music object có nhiều track, gộp các track lại vào track đầu tiên của Music object để có dạng ma trận chuẩn của pianoroll (128 x Timestep). 

    Input -- Music object

    Output -- Trả về Music object chỉ có đúng một track biểu diễn toàn bộ các nốt nhạc của bản nhạc
    '''

    i = 1
    while i < len(score.tracks):
        tmp = score.tracks[i].notes
        score.tracks[0].notes.extend(tmp)
        score.tracks.pop(i)

def inferBeat(score: muspy.Music):
    '''
    Tạo một vector kích thước Timestep. Các giá trị cho biết tại thời điểm timestep có phải là nhịp đầu tiên của một ô nhịp không. Giá trị thuộc kiểu nhị phân (0, 1).

    Input -- Music object

    Output -- Trả về vector như trên
    '''

    numTimestep = max(note.end for note in (score.tracks[0].notes) ) + 1 #For compatible with array returned by from/to_pianoroll_representation
    listFirstBeat = np.array([0] * numTimestep, bool)
    listTimeBeat = score.barlines
    
    for it in listTimeBeat:
        listFirstBeat[ it.time ] = 1

    return listFirstBeat

def inferTempo(score: muspy.Music):
    '''
    Tạo một vector kích thước Timestep. Các giá trị cho biết tại thời điểm timestep tempo của giai điệu là bao nhiêu

    Input -- Music object

    Output -- Trả về vector như trên
    '''

    numTimestep = max(note.end for note in (score.tracks[0].notes) ) + 1 #For compatible with array returned by from/to_pianoroll_representation
    listTimeTempo = score.tempos
    listTempo = np.array([0] * numTimestep, dtype = np.int16)

    for it in listTimeTempo:
        listTempo[ it.time ] = it.qpm
    for i in range(0, numTimestep):
        if listTempo[i] == 0:
            listTempo[i] = listTempo[i - 1]

    return listTempo

def inferListTimeTempo(pianoroll: np.ndarray):

    '''
    Lấy danh sách các tempo changing event từ hàng cuối cùng của ma trận pianoroll

    Input -- pianoroll: ma trận pianoroll của một bản nhạc chứa hàng tempo event tại hàng số 129

    Output -- danh sách các đối tượng Tempo của muspy
    '''

    listTimeTempo = []
    listTimeTempo.append( muspy.Tempo(0, pianoroll[0, 129] if pianoroll[0, 129] != 0 else 120) )

    for i in range(1, pianoroll.shape[0]):
        if pianoroll[i, 129] != pianoroll[i-1, 129]:
            listTimeTempo.append( muspy.Tempo(i, pianoroll[i, 129]) )

    return listTimeTempo

def inferListTimeBeat(pianoroll: np.ndarray):
    '''
    Lấy danh sách các timestep bắt đầu một ô nhịp

    Input -- pianoroll: Ma trận pianoroll của một bản nhạc, chứa các first beat tại hàng số 128

    Output -- danh sách các đối tượng Barline của muspy
    '''

    listTime = []
    for i in range(0, pianoroll.shape[0]):
        if pianoroll[i, 128] == 1:
            listTime.append( muspy.Barline(i) )

    return listTime

def inferKey(score):
    '''
    Đoán giọng của bản nhạc (để xử dụng cho thao tác biến đổi dataset)

    Input -- Music object
    Output -- Trả về đối tượng Key của thư viện music21 chứa thông tin của khóa
    '''

    muspy.write_midi('tmp.mid', score, backend = "pretty_midi")

    piece = music21.converter.parse('tmp.mid')
    key = piece.analyze('key')

    #os.remove('tmp.mid')

    return key

def calculateEmotion(oEmotion, valence = None, activation = None):
    '''
    Tính toán nhãn cảm xúc thông qua điều chỉnh trục valence và activation

    Input -- oEmotion: nhãn cảm xúc đã quy ước
            valence: lượng valence mới 
            activation: lượng activation mới
    
    Output -- Trả về nhãn cảm xúc mới sau khi điều chỉnh
    '''
    new_point = ()

    new_point += (valence, ) if valence is not None else (Config.emotion[oEmotion][0], )
    new_point += (activation, ) if activation is not None else (Config.emotion[oEmotion][1], )

    return Config.emotion.inverse[new_point]

def openJSONFile(path):
    '''
    Đọc một file JSON và trả về một dictionary tương ứng

    Input -- Đường dẫn tới file JSON

    Output -- Dictionary biểu diễn giá trị của file JSON vừa đọc
    '''

    with open(path, 'r') as f:
        dic = json.load(f)

    return dic

def writeJSONFile(path, dic):
    '''
    Chuyển một dictionary về kiểu JSON và viết vào file

    Input -- path: đường dẫn file JSON
            dic: dictionary biểu diễn dữ liệu cần ghi

    Output -- tạo một file JSON tại đường dẫn path
    '''

    with open(path, 'w') as f:
        f.write( json.dumps(dic, indent = 4) )

def createMetadataNormalized(path):
    '''
    Tạo một cấu trúc và file JSON chứa metadata của các bản nhạc đã được chuẩn hóa trong thư mục Dataset_normalized

    Input -- path: đường dẫn file JSON

    Output -- trả về một dictionary biểu diễn file JSON được tạo
    '''

    if not os.path.exists(path):

        dic = {
            "id": None,
            "name": None,
            "style": None,
            "emotion": None,
            "key": None
        }
        writeJSONFile(path, dic)

    return openJSONFile(path)

def augmentateTempo(pianoroll, percent, dic):
    '''
    Augmentate dataset bằng cách tăng/giảm tempo của toàn bộ bản nhạc thành một lượng mới bằng percent% lượng cũ của tempo

    Input -- pianoroll: pianoroll chứa bản nhạc cần điều chỉnh tempo
            -- percent: lượng mới của tempo
            -- dic: dictionary chứa metadata của pianoroll này

    Output -- Trả về một pianoroll chứa bản nhạc đã được điều chỉnh tempo, tạo các file JSON và pianoroll trong thư mục Dataset_normalized
    '''
    global id_pianoroll

    new_pianoroll = pianoroll
    new_dic = dic
    print(new_pianoroll[:, 129])
    new_pianoroll[:, 129] = new_pianoroll[:, 129] * float(percent/100)
    print(new_pianoroll[:, 129])
    print(new_pianoroll.base)
    np.save("Dataset\\Dataset_normalized\\" + str(id_pianoroll) + '.npy', new_pianoroll)
    
    new_dic['id'] = id_pianoroll
    if percent > 100:
        new_dic['emotion'] = calculateEmotion(new_dic['emotion'], activation = 1)
    else:
        new_dic['emotion'] = calculateEmotion(new_dic['emotion'], activation = -1)

    writeJSONFile("Dataset\\Dataset_normalized\\" + str(id_pianoroll) + '.json', new_dic)
    id_pianoroll += 1

    return new_pianoroll

def augmentateKey(score: muspy.Music, dic, key):
    '''
    Augmentate dataset bằng cách chỉnh giọng của bản nhạc sao cho thu được đủ 12 giọng trưởng hoặc thứ tương ứng

    Input -- score: Music object cần điều chỉnh giọng
            dic: dictionary chứa metadata của bản nhạc này
            key: giọng gốc của bản nhạc

    Output -- Tạo ra các pianoroll của bản nhạc với 11 giọng khác nhau (không bao gồm giọng của bản gốc)
    '''
    global id_pianoroll

    old_key = dic['key']
    new_score = score.deepcopy()
    listKey = Config.majorkey if key.mode == "major" else Config.minorkey
    transposeLow = False #If tranpose to higher pitch failed, transpose to lower pitch instead

    for i in range(1, 12):

        print("Tranpose {}".format(i))

        new_dic = dic
        new_dic['id'] = id_pianoroll
        muspy.transpose(new_score, i) if not transposeLow else muspy.transpose(new_score, -(12 - i)) 
        new_dic['key'] = listKey.inverse[ ( (listKey[old_key][0] + i) % 12, 0) ]
        new_dic['emotion'] = Config.keyEmotion[ new_dic['key'] ]

        #If we can not transpose to higer pitch, we reserve the transpose direction in this turn and following turn
        try:
            pianoroll = createPianoRepresentation(new_score)
        except:
            print("Transpose failed. Reserved")
            if transposeLow == False:
                transposeLow = True
                i -= 1
            else:
                break #If we continue fail, stop transpose augmentation

        np.save("Dataset\Dataset_normalized\\" + str(id_pianoroll) + '.npy', pianoroll)
        writeJSONFile("Dataset\\Dataset_normalized\\" + str(id_pianoroll) + '.json', new_dic)
        id_pianoroll += 1

        new_score = score.deepcopy() #Restore to original key

def createSamples(pianoroll: np.ndarray, num_bar, path, id):
    '''
    Tạo các sample từ các pianoroll với độ dài mỗi sample là num_bar. Khoảng dịch chuyển của hai sample liền kề nhau thuộc cùng một pianoroll hơn kém nhau 1 bar

    Input -- pianoroll: pianoroll cần tạo các samples
            numbar: độ dài một sample, tính theo số ô nhịp
            path: đường dẫn để lưu các sample
            id: số thứ tự của file pianoroll trong thư mục Dataset_normalized

    Output -- num_bar: độ dài của một sample
    '''

    global id_sample

    dic = openJSONFile('Dataset\Samples\link.json')

    id_str = str(id)
    listTimeFirstBeat = inferListTimeBeat(pianoroll)
    i = 0
    numBar = len(listTimeFirstBeat)

    while i + num_bar < numBar:
        start = listTimeFirstBeat[i].time
        end = listTimeFirstBeat[i + num_bar].time
        sample = pianoroll[start : end - 1, ::]
        np.save(path + '\\' + str(id_sample) + '.npy', sample)
        dic.update( {str(id_sample): id_str} )
        id_sample += 1
        i += 1

    sample = pianoroll[ listTimeFirstBeat[i].time :, ::]
    np.save(path + '\\' + str(id_sample) + '.npy', sample)
    dic.update( {str(id_sample): id_str} )
    id_sample += 1

    writeJSONFile('Dataset\Samples\link.json', dic)

if __name__ == "__main__":

    #createMetadaOriginalDataset('Dataset\Dataset_original\Classical_dataset\mond_3.mid')
    normalizedDataset('Dataset\Dataset_original\Classical_dataset\mond_3.mid')
    writeJSONFile('Dataset\Samples\link.json', {})
    createSamples(np.load('Dataset\Dataset_normalized\\1.npy'), 2, 'Dataset\Samples', 1)

    pianoroll = np.load("Dataset\Dataset_normalized\\13.npy")
    score = muspy.from_pianoroll_representation(pianoroll)
    score.tempos = inferListTimeTempo(pianoroll)
    inferKey(score)

