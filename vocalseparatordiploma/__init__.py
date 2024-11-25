import vocalseparatordiploma.preprocessing as preprocessing
import vocalseparatordiploma.prediction as prediction
import vocalseparatordiploma.postprocessing as postprocessing


def separate(mix_path: str, vocals_path: str, instrumental_path: str):
    """
    main function of the system, splits mix file into two files(vocals, instrumental)
    :return:
    """
    preprocessing.read_track()  # read signal from file
    preprocessing.compute_stft()  # compute stft
    preprocessing.generate_windows()  # get input data from stft

    prediction.predict()  # or predict_mocked, get model output

    postprocessing.combine_prediction_outputs()  # get final mask
    postprocessing.apply_mask()  # apply mask on original stft, returns vocal and instrumental stft
    postprocessing.compute_inverse_stft()  # compute signal
    postprocessing.write_track()  # save vocals track
    postprocessing.write_track()  # save instrumental track
