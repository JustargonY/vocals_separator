import vocalseparatordiploma.preprocessing as preprocessing
import vocalseparatordiploma.prediction as prediction
import vocalseparatordiploma.postprocessing as postprocessing


def separate(mix_path: str, vocals_path: str, instrumental_path: str):
    """
    main function of the system, splits mix file into two files(vocals, instrumental)
    """
    sr, audio = preprocessing.read_track(mix_path)  # read signal from file
    frequencies, times, zxx_l, zxx_r = preprocessing.compute_stft(audio, sr)  # compute stft
    # preprocessing.generate_windows()  # get input data from stft

    # prediction.predict()  # or predict_mocked, get model output

    # mask_l = postprocessing.combine_prediction_outputs()  # get final mask
    # mask_r = postprocessing.combine_prediction_outputs()  # get final mask
    mask_l = postprocessing.get_mocked_mask(zxx_l)
    mask_r = postprocessing.get_mocked_mask(zxx_r)
    zxx_vocals_l, zxx_instr_l = postprocessing.apply_mask(zxx_l, mask_l)  # apply mask on original stft, returns vocal and instrumental stft
    zxx_vocals_r, zxx_instr_r = postprocessing.apply_mask(zxx_r, mask_r)  # apply mask on original stft, returns vocal and instrumental stft
    vocals = postprocessing.compute_inverse_stft(sr, zxx_vocals_l, zxx_vocals_r)  # compute signal
    instr = postprocessing.compute_inverse_stft(sr, zxx_instr_l, zxx_instr_r)

    postprocessing.write_track(vocals_path, sr, vocals)  # save vocals track
    postprocessing.write_track(instrumental_path, sr, instr)  # save instrumental track

