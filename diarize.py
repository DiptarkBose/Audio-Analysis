from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
from spectralcluster import SpectralClusterer
from pydub import AudioSegment
import os

#give the file path to your audio file
audio_file_path = 'two_friends.wav'
wav_fpath = Path(audio_file_path)

wav = preprocess_wav(wav_fpath)
encoder = VoiceEncoder("cpu")
_, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)
#print(cont_embeds.shape)

clusterer = SpectralClusterer(
    min_clusters=2,
    max_clusters=10,
    p_percentile=0.90,
    gaussian_blur_sigma=1)

labels = clusterer.predict(cont_embeds)

def create_labelling(labels,wav_splits):
    from resemblyzer import sampling_rate
    times = [((s.start + s.stop) / 2) / sampling_rate for s in wav_splits]
    labelling = []
    start_time = 0

    for i,time in enumerate(times):
        if i>0 and labels[i]!=labels[i-1]:
            temp = [str(labels[i-1]),start_time,time]
            labelling.append(tuple(temp))
            start_time = time
        if i==len(times)-1:
            temp = [str(labels[i]),start_time,time]
            labelling.append(tuple(temp))

    return labelling
  
labelling = create_labelling(labels,wav_splits)

#print(labelling)
current_path = os.getcwd()
speaker0_path = current_path+"/speaker0"
speaker1_path = current_path+"/speaker1"

for speech_marker in labelling:
	t1 = speech_marker[1] * 1000 #Works in milliseconds
	t2 = speech_marker[2] * 1000
	newAudio = AudioSegment.from_wav(audio_file_path)
	newAudio = newAudio[t1:t2]
	newFileName = str(speech_marker[1])+"_"+str(speech_marker[2])
	print("Speaker-"+speech_marker[0]+" spoke from "+str(speech_marker[1])+" to "+str(speech_marker[2]))
	print
	if(speech_marker[0]=="0"):
		newFileName = newFileName + "_speaker0.wav"
		os.chdir(speaker0_path)
		newAudio.export(newFileName, format="wav")
		os.chdir("..")
	else:
		newFileName = newFileName + "_speaker1.wav"
		os.chdir(speaker1_path)
		newAudio.export(newFileName, format="wav")
		os.chdir("..")
