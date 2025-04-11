import numpy as np
import pywt


def time_feature(data):
    # 最大值
    Max = np.max(data)

    #  最小值
    Min = np.min(data)

    #  峰峰值 peak-peak
    PP = Max - Min

    #  均值
    Mean = np.mean(data)

    #  整流平均值 rectified mean value
    RMV = np.mean(abs(data))

    #  均方根 root mean square
    RMS = np.sqrt(np.mean(data ** 2))

    #  方根幅值 root square amplitude
    RSA = np.sqrt(np.mean(data ** 2))

    #  方差
    Var = np.var(data)

    #  标准差
    Std = np.std(data)

    #  峭度 kurtosis
    Kur = np.sum((data - Mean) ** 4) / (len(data) * Std ** 4)

    #  偏度 skewness
    Skew = np.sum((data - Mean) ** 3) / (len(data) * Std ** 3)

    #  裕度因子 margin factor
    MF = Max / RSA

    #  峰值因子  peak factor
    PF = Max / RMS

    #  脉冲因子 impulse factor
    IF = Max / RMV

    #  波形因子 shape factor
    SF = RMS / RMV

    #  能量
    E = np.sum(data ** 2)

    feature_list = [np.round(Max, 3), np.round(Min, 3), np.round(PP, 3),
                    np.round(Mean, 3), np.round(RMV, 3), np.round(RMS, 3),
                    np.round(RSA, 3), np.round(Var, 3), np.round(Std, 3),
                    np.round(Kur, 3), np.round(Skew, 3), np.round(MF, 3),
                    np.round(PF, 3), np.round(IF, 3), np.round(SF, 3),
                    np.round(E, 3)
                    ]
    feature_list = np.array(feature_list)
    return feature_list


def wavelet_feature(data):
    data = np.squeeze(data)
    wp = pywt.WaveletPacket(data=data, wavelet='db3', mode='symmetric', maxlevel=2)
    new_wp = pywt.WaveletPacket(data=None, wavelet='db3', mode='symmetric', maxlevel=2)

    new_wp['aa'] = wp['aa']
    LL = new_wp.reconstruct(update=False)

    del (new_wp['aa'])
    new_wp['ad'] = wp['ad']
    LH = new_wp.reconstruct(update=False)

    del (new_wp['a'])
    new_wp['da'] = wp['da']
    HL = new_wp.reconstruct(update=False)

    del (new_wp['da'])
    new_wp['dd'] = wp['dd']
    HH = new_wp.reconstruct(update=False)

    LLPE = np.sum(LL * LL)
    LHPE = np.sum(LH * LH)
    HLPE = np.sum(HL * HL)
    HHPE = np.sum(HH * HH)

    E = [LLPE, LHPE, HLPE, HHPE]
    P = E / np.sum(E)

    W_Entropy = -np.sum(np.log2(P) * P)
    WIQ = np.sum(P * E)

    feature_list = [np.round(LLPE, 3), np.round(LHPE, 3),
                    np.round(HLPE, 3), np.round(HHPE, 3),
                    np.round(W_Entropy, 3), np.round(WIQ, 3),
                    ]
    feature_list = np.array(feature_list)
    return feature_list
