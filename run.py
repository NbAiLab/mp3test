from transformers import pipeline
import time
import glob
from pydub import AudioSegment

#pipe = pipeline(model="NbAiLab/nb-wav2vec2-300m-bokmaal",device=0)
pipe = pipeline(model="NbAiLab/nb-wav2vec2-1b-bokmaal", return_timestamps="word",device=0)

files = glob.glob('mp3test/*.mp3')[:10]

totallength = 0
for f in files:
    audio = AudioSegment.from_mp3(f)
    length = (len(audio)/1000)
    totallength += length
    print(f"File is {length} seconds")
print(f"\n** Total is {totallength}\n")

start = time.perf_counter()
output = pipe(files)
print(output)
end = time.perf_counter()
elapsed_time = end - start

print(f"\n* The {len(files)} files were {totallength} seconds long. It was processed in {elapsed_time:.2f} seconds. Factor was: {totallength/elapsed_time}\n")

exit()

for f in files:
    audio = AudioSegment.from_mp3(f)
    
    start = time.perf_counter()
    output = pipe(f)
    print(output)
    end = time.perf_counter()
    elapsed_time = end - start
    print(f"\n* The file is {len(audio)/1000} seconds long. It was processed in {elapsed_time:.2f} seconds. Factor was: {(len(audio)/1000)/elapsed_time}\n")
