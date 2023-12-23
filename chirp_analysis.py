# very basic implementation of analyzing the frequency response of a recorded log-swept sine chirp using Farina method ("Simultaneous Measurement of Impulse Response and Distortion", Angelo Farina, 2000)
# only isolates the fundamental frequency. Could be extended to measure harmonic distortion by windowing out harmonic impulses as described in the paper

# everything available in standard Anaconda distribution at time of writing
import sys
import numpy as np
import math
from scipy.io import wavfile
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from scipy import interpolate

if len(sys.argv) == 1:
    print("usage: python chirp_analysis.py <response file> <csv_out>\n"
          "calulates the frequency response of the response file against chirp parameters defined in the script\n"
          "if no CSV file path is given, plots the frequency response")
    exit()


def main():
    # chirp parameters
    start_freq = 100 # frequencies in Hz
    stop_freq = 20000
    sample_rate = 48000
    chirp_length = int(1.0*sample_rate) # chirp length in samples
    
    
    # analysis parameters
    pre_sweep = int(0.05*sample_rate) # pre/post sweep is time included before/after analysis window. Helps avoid issues with slight timing issues around abrupt start/end of chirp
    post_sweep = int(0.05*sample_rate)
    ir_start = int(0.001 * sample_rate) # length of time in samples to include in the impulse response analysis before the detected impulse response start. Setting this to 0 risks cutting off part of the impulse response before t=0 that may result from phase distortion or slight time misalignment
    ir_stop = int(0.05 * sample_rate) # this is the typical impulse response window length, as typically used in time-gated frequency response measurements. Shorter windows reduce measurement noise, at the cost of reduced sensitivity and accuracy at low frequencies
    num_points = int(24 * math.log(stop_freq/start_freq,2)) + 1 # number of frequency points to use for CSV output file. 24 points per octave is usually a good medium resolution for acoustic frequency response

    response_file = sys.argv[1]
    bit_depth = 16 # bit depth of the input file. Scipy has a lot of issues reading perfectly normal audio files, 16 or 32-bit WAV files tend to work best. Look into audio2numpy for better compatibility?

    csv_out = sys.argv[2] if len(sys.argv) > 2 else False

    stimulus = logchirp(start_freq, stop_freq, chirp_length/sample_rate, sample_rate)
    stimulus = np.concatenate((np.zeros(pre_sweep), stimulus, np.zeros(post_sweep)))
    #plt.plot(stimulus)
    #plt.show()
    
    response = wavfile.read(response_file)[1]
    response = response / 2**bit_depth # normalize integer values to [-1:1]
    #plt.plot(response)
    #plt.show()
    
    
    
    # find the relative delay between the stimulus and response signals and trim the response to match
    # create padded version of stimulus (everything will break if input response file is shorter than chirp length + pre/post sweep)
    padded_stimulus = np.append(stimulus, np.zeros(len(response) - len(stimulus))) # this may be unneccessary, delay may be able to be determined as delta between argmax and length of first array in cross correlation (or something like that). Need to experiment to confirm
    
    # compute the cross correlation between the stimulus and response signals and determine the relative delay between them
    correlation = fftconvolve(padded_stimulus, response[::-1]) # convolve stimulus with reversed response
    delay = int(len(correlation) / 2) - np.argmax(correlation) # relative delay is the delta between the peak of the cross correlation and the midpoint of the cross correlation
    #print(delay)
    #plt.plot(correlation)
    #plt.show()
    
    # trim response signal
    response = response[delay:delay+len(stimulus)]
    #plt.plot(response)
    #plt.show()
    
    
    
    # calculate the frequency response
    fr = fft(response)/fft(stimulus) # this is the complex frequency response, including negative half of frequency spectrum
    # 20*log10(fr) for magnitude in dB. Assuming stimulus signal peaks at +/-1.0, magnitude unit is dBFS
    # need to verify, but unwrapped phase response should be atan2(fr)
    #plt.plot(20*np.log10(np.abs(fr)))
    #plt.show()
    
    # convert frequency response to impulse response
    ir = ifft(fr)
    #plt.plot(ir)
    #plt.show()
    
    # generate a mask to window the fundamental peak out of the impulse response
    window = np.concatenate((np.ones(ir_start+ir_stop), np.zeros(len(stimulus) - ir_start - ir_stop))) # rectangular window over the desired range, plus zero padding. Fading window in/out with half Hann windows or similar would be better (allow you to use shorter time gates without artefacts), but rectangular is fine, assuming you have 1) a little silence before/after the captured chirp (pre/post_sweep), 2) cross correlation results in good time alignment and trimming, and 3) the overall response SNR is high (~30+dB is probably fine)
    window = np.roll(window, -ir_start) # circularly shift the window to align with the impulse response at t=0
    #plt.plot(window)
    #plt.show()
    
    # apply window to impulse response
    ir = ir * window
    #plt.plot(ir)
    #plt.show()
    
    # calculate the new windowed frequency response
    fr = fft(ir)
    #plt.plot(20*np.log10(np.abs(fr)))
    #plt.show()
    
    # plot frequency response or output CSV
    fr_freqs = fftfreq(len(stimulus), 1/sample_rate) # calculate the center frequency for each FFT bin
    fr_freqs = fr_freqs[1:int(len(fr)/2)-1] # trim to only positive frequencies. Avoids line across graph and issues with interpolation in CSV output
    fr = fr[1:int(len(fr)/2)-1]
    
    if csv_out: # write CSV file
        fr_interp = interpolate.interp1d(np.log(fr_freqs),20*np.log10(np.abs(fr))) # linear interpolation of frequency response over a logarithmic frequency scale
        csv_freqs = np.round(start_freq * ((stop_freq/start_freq)**(1/(num_points-1)))**np.arange(num_points)) # generate proportionally spaced frequency points
        
        f = open(csv_out,'w')
        f.write('Frequency (Hz),dBFS\n') # write header
        for freq in csv_freqs:
            f.write(str(freq) + ',' + str(fr_interp(np.log(freq))) + '\n') # write each point
        f.close()
        
    else: # display frequency response graph
        plt.plot(fr_freqs, 20*np.log10(np.abs(fr))) # plot frequency response in dBFS, format graph
        plt.xlim(start_freq, stop_freq)
        plt.xscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('dBFS')
        plt.title('Frequency Response')
        plt.show()









def logchirp(start_freq, stop_freq, length, sample_rate):
    # generate the samples of a log-swept sine chirp signal given
    # start and stop frequencies in Hz
    # length in seconds
    # sample rate in Hz
    t = np.arange(length*sample_rate)/sample_rate
    f_scalar = (2*math.pi*start_freq*length)/math.log(stop_freq/start_freq)
    f_exp = (math.log(stop_freq/start_freq)*t)/length
    return np.sin(f_scalar * (np.exp(f_exp)-1))




main()