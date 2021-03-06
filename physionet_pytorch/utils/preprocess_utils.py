from scipy.signal import butter, sosfilt

class HighpassFilter(object):

    def __init__(self, fs, fc, order):
        nyq = 0.5 * fs
        norm_fc = fc / nyq
        self.sos = butter(order, norm_fc, btype='highpass', output='sos')

    def __call__(self, sample):
        for ch in sample.shape[0]:
            sample[ch, :] = sosfilt(self.sos, sample[ch, :])
        return sample


class BandpassFilter(object):

    def __init__(self, fs, fc_low, fc_high, order):
        nyq = 0.5 * fs
        norm_fc_low = fc_low / nyq
        norm_fc_high = fc_high / nyq
        self.sos = butter(order, [norm_fc_low, norm_fc_high], btype='bandpass', output='sos')

    def __call__(self, sample):
        for ch in sample.shape[0]:
            sample[ch, :] = sosfilt(self.sos, sample[ch, :])
        return sample


class Identity(object):

    def __call__(self, sample):
        return sample


class TimeWindowPostCue(object):

    def __init__(self, fs, t1_factor, t2_factor):
        self.t1 = int(t1_factor * fs)
        self.t2 = int(t2_factor * fs)

    def __call__(self, sample):
        return sample[:, :, self.t1:self.t2]


class ReshapeTensor(object):
    def __call__(self, sample):
        return sample.view(1, sample.shape[0], sample.shape[1])
