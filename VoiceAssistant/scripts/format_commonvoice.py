import os
import pandas as pd
from pydub import AudioSegment
from pydub.utils import make_chunks
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

def main(args):
    df = pd.read_csv(args.file_name), sep='\t')
    print(df.head())
    print('total data size:', len(df))

    def chunk_and_save(file):
        path = os.path.join(args.data_path, file)
        audio = AudioSegment.from_file(path)
        length = args.seconds * 1000 # this is in miliseconds
        chunks = make_chunks(audio, length)
        names = []
        for i, chunk in enumerate(chunks):
            _name = file.split(".mp3")[0] + ".wav"
            name = "{}_{}".format(i, _name)
            wav_path = os.path.join(args.save_path, name)
            chunk.export(wav_path, format="wav")
        return names
    df.path.parallel_apply(lambda x: chunk_and_save(x))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
    Script to collect data for wake word training..

    To record environment sound run set seconds to None. This will
    record indefinitely until ctrl + c.

    To record for a set amount of time set seconds to whatever you want.

    To record interactively (usually for recording your own wake words N times)
    use --interactive mode.
    ''')
    parser.add_argument('--sample_rate', type=int, default=8000,
                        help='the sample_rate to record at')
    parser.add_argument('--seconds', type=int, default=None,
                        help='if set to None, then will record forever until keyboard interrupt')
    parser.add_argument('--data_path', type=str, default=None, required=True,
                        help='full path to data. i.e. /to/path/clips/')
    parser.add_argument('--save_path', type=str, default=None, required=True,
                        help='full path to to save data. i.e. /to/path/saved_clips/')

    args = parser.parse_args()

    main(args)
