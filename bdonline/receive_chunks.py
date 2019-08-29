from pylsl import StreamInlet, resolve_stream
import time
# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0], max_chunklen=200)

    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)
samples_pulled = 0
start_time = time.time()
while samples_pulled < 200000:
    chunk, timestamps = inlet.pull_chunk(timeout=0.05)
    samples_pulled += len(chunk)
end_time = time.time()

print('it took {} secs'. format(end_time -start_time))