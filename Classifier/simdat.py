# test
import numpy as np
from matplotlib import pyplot as plt


def noise(length: int = 100, amplitude: float = None, *, noise_type: str = None):
    """
    :param amplitude:
    :param length:
    :param noise_type: Uniform or Normal
    :return:
    """
    amplitude = 0.1 if amplitude is None else amplitude
    if (noise_type is None) or ("uniform" == noise_type):
        return np.random.uniform(-amplitude, amplitude, (length,))
    elif "normal" == noise_type:
        return np.random.normal(0, amplitude, (length, ))
    else:
        raise ValueError("Unsupported noise type. Use 'normal' or 'uniform'.")


def sine(frequency: float, amplitude: float, xs: np.ndarray):
    """
    :param frequency:
    :param amplitude:
    :param xs:
    :return:
    """
    return amplitude*np.sin(frequency*np.pi*xs)


def spikes(length: int = 100, spike_width: int = None, spike_height: float = None,
           probability_at_end: float = None, return_spike_locations: bool = False):
    """
    First determines locations where to create spikes, then makes spikes at those locations.
    :param length:
    :param spike_width:
    :param spike_height:
    :param probability_at_end:
    :param return_spike_locations:
    :return:
    """
    spike_width = 9 if spike_width is None else spike_width
    spike_height = 10. if spike_height is None else spike_height
    probability_at_end = 0.01 if probability_at_end is None else probability_at_end
    spike_middle = (spike_width - 1) / 2
    new_length = length - spike_width
    probability_array = np.linspace(0, probability_at_end, new_length)
    random_array = np.random.rand(new_length)
    spike_locations = random_array < probability_array
    ans = np.zeros(length)
    single_spike = np.array([spike_height/spike_middle*i*(2-i/spike_middle) for i in range(spike_width)])
    for i, val in enumerate(spike_locations):
        if val:
            ans[i:i+spike_width] += single_spike
    if return_spike_locations:
        return ans, spike_locations
    return ans


def composite_signal(length: int, sine_tuples=None, *, noise_amplitude: float = None,
                     spike_width: int = None, spike_height: float = None,
                     return_spike_locations: bool = False):
    """
    Makes a composite signal of both sines, noises and spikes
    :param length:
    :param sine_tuples:
    :param noise_amplitude:
    :param spike_width:
    :param spike_height:
    :param return_spike_locations:
    :return:
    """
    sine_tuples = tuple() if sine_tuples is None else sine_tuples
    xs = np.arange(0, length, 1)
    array_list = [sine(freq, amp, xs) for freq, amp in sine_tuples]
    array_list.append(noise(length, noise_amplitude, noise_type="uniform"))
    if not return_spike_locations:
        array_list.append(spikes(length, spike_width, spike_height))
        return sum(array_list)
    spikes_array, spike_locations = spikes(length, spike_width, spike_height)
    array_list.append(spikes_array)
    return sum(array_list), spike_locations


#test1 = composite_signal(1000, ((0.1, 2), (0.19, 1)), noise_amplitude=0.8)
# plt.plot(test1)
# plt.show()



