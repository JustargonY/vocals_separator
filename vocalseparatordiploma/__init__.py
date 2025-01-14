from vocalseparatordiploma import preprocessing
from vocalseparatordiploma import prediction
from vocalseparatordiploma import postprocessing


def separate(mix_path: str, vocals_path: str, instrumental_path: str):
    """
    main function of the system, splits mix file into two files(vocals, instrumental)
    """
    audio = preprocessing.read_track(mix_path)  # read signal from file

    vocals, instr = prediction.predict_signal(audio)

    postprocessing.write_track(vocals_path, vocals)  # save vocals track
    postprocessing.write_track(instrumental_path, instr)  # save instrumental track
