---
layout: post
title: Digital Filter Design
categories: []
tags: [DSP]
mathjax: true
show_meta: true
---

Consider an audio [signal](https://www.dropbox.com/s/yxfwuc8gvet90g5/signal_with_noise.wav?dl=1). The speech interferes with high-pitch noise. Preliminary inspection of the signal waveform does not reveal anything particular about this recording. 

```matlab
// Read signal
[s, Fs, _] = wavread("7cef8230.wav")
// Take the first channel
s = s(1, :)
// Plot
plot(s)
xlabel("Time, n", 'fontsize', 2)
ylabel("Amplitude", 'fontsize', 2)
title("Signal with noise in time domain", 'fontsize', 3)
xs2png(0, "signal_with_noise_time.png")
// close() // close immediately after saving
```
<p align="center">
<img src="https://i.imgur.com/3QmyWqF.png" width="500">
</p>

Working with signals in time domain (unless these signals are very simple) is not very straihtforward. However, it appears that many signals are separable when projected into the frequency domain. The tool that allows to compute the frequency spectrum is Fourier transform.
```matlab
// Plot spectrum
// calculate frequencies
s_len = length(s)
frequencies = (0:s_len-1)/s_len * Fs;
// plot
plot2d("nl", frequencies, abs(fft(s)), color("blue"))
xlabel("Frequency component, n", 'fontsize', 2)
ylabel("Freq Amplitude", 'fontsize', 2)
title("Signal with noise in frequency domain", 'fontsize', 3)
xs2png(0, "signal_with_noise_freq.png")
// close() // close immediately after saving
```
<p align="center">
<img src="https://i.imgur.com/kYnbYUl.png"  width="500">
</p>

Since the signal is sampled at the rate of $$F_s=44100Hz$$, the frequency spectrum is symmetric over $$\frac{F_s}{2}=22050Hz$$. The speech signal is usually located at the frequencies below $$5kHz$$. It is easy to notice the presence of high-frequency noise around $$10kHz$$. Since the speech is located at lower frequencies, it is possible to separate it from the noise using a lowpass filter. The lowpass filter is such a filter that allows only low-frequency components to pass through. The filter is defined with a series of coefficients $$(h_{low})_n$$.

The goal of the FIR filter design is to identify the set of coefficients for the filter. Here, we are going to discuss the Window filter design method that, in general, consist of the following steps:
1. Specify the desired frequency response
2. Compute impulse response in the temporal domain
3. Apply a window function
4. Evaluate the result

The goal is to design a causal real-valued FIR filter. In order to achieve this, we will need to perform extra steps that will be discussed below. 

### Desired frequency response

The frequency response of the ideal low pass filter $$H(e^{j \omega})$$ has rectangular form: 
$$
    H(e^{j \omega})=
    \begin{cases}
      1, & \text{if}\  \vert \omega \vert  < \omega_{cutoff} \\
      g_{stop}, & \text{otherwise}
    \end{cases}
$$
where $$g_{stop}$$ is the gain of the filter in the stop band. Passband frequency range (frequencies that pass through the filter) is specified by the range $$[-\omega_{cutoff}, \omega_{cutoff}]$$. There are two main points to keep in mind when designing FIR filter in the frequency domain:
- The function $$H(e^{j \omega}), \omega \in [-\pi, \pi]$$ is periodic with the period $$2 \pi$$
- Most of the libraries that implement DFT assume $$\omega \in [0, 2\pi]$$

Hence, the definition above can be rewritten:
$$
    H(e^{j \omega})=
    \begin{cases}
      1, & \text{if}\ \omega \in [0, \omega_{cutoff}] \text{ or } \omega \in [2\pi - \omega_{cutoff}, 2\pi] \\
      g_{stop}, & \text{otherwise}
    \end{cases}
$$
Using this last form for the ideal low-pass filter one can construct its frequency response:
```matlab
// N: integer length of FIR filter
// cutoff: fraction of Fs, at which frequencies are stopped
// stop_value: the value for frequencies in the stop band 
//    (after cutoff frequency)
// return: frequency representation of an ideal 
//   low pass FIR filter of length N+1 if N is even
//   or N if N is odd
function H = ideal_lowpass(N, cutoff, stop_value)
    N = (N - modulo(N,2)) / 2 
    cutoff = floor(2 * N * cutoff)
    H = ones(1, N) * stop_value
    H(1,1:cutoff) = 1.
    // need to make N odd
    H = [1. H flipdim(H, 2)]   // <---- line 14
endfunction


// Plot ideal lowpass freq response
// calculate lowpass
// Filter will have length 257
H_l = ideal_lowpass(256, 0.15, 0.);
// calculate frequencies
h_len = length(H_l)
frequencies = (0:h_len-1)/h_len * Fs;
// plot
plot2d("nn", frequencies, H_l, color("blue"))
xlabel("Frequency, Hz", 'fontsize', 2)
ylabel("Freq amplitude", 'fontsize', 2)
title("Frequency response of ideal low-pass filter", 'fontsize', 3)
xs2png(0, "ideal_lowpass_freq.png")
// close() // close immediately after saving
```


Notice that on the line `14`, we mirror the response and add `1.` at $$0^{th}$$ frequency. We need to mirror the response to ensure the frequency response is [Hermitian](https://en.wikipedia.org/wiki/Hermitian_function), and its inverse Fourier transform is real-valued. We add `1.` at 0th frequency, first, to ensure the length of the response is odd and to let our lowpass filter to pass constant signals through. The frequency response has the following form:
<p align="center">
<img src="https://i.imgur.com/Xkvp0O6.png" width="500">
</p>

### Computing Impulse Response
The impulse response after projection into the time domain is
```matlab
// Compute impulse response
// project into temporal domain
// imaginary part should be close to 0
h_l = real(ifft(H_l))
plot2d('nn', 0:length(h_l)-1, h_l, color("blue"))
xlabel("Time, n", 'fontsize', 2)
ylabel("Amplitude", 'fontsize', 2)
title("Impulse response of ideal low-pass filter", 'fontsize', 3)
xs2png(gcf(), "ideal_lowpass_time.png")
```

<p align="center">
<img src="https://i.imgur.com/sO2uTGz.png" width="500">
</p>

From the waveform above, one can notice several things:
- Impulse response is $$sinc$$ function
- Impulse response never decays to zero
- Impulse response is looped due to cyclic convolution of DFT

These points should be addressed to make this FIR filter useful.

Currently, the frequency response of the filter has only real part, and imaginary part is equal to zero. Such filters are called zero-phase filter, and in general such filters are not causal. To address this problem, one needs to introduce linear phase to the frequency response. This will result in the shift of the frequency response
$$
\tilde{H}(e^{j \omega}) = {H}(e^{j \omega}) \cdot e^{-j\omega k} 
$$
where $$\tilde{H}(e^{j \omega})$$ is the impulse response shifted by $$k \in [0, N-1]$$.

<p align="center">
<img src="https://i.imgur.com/n5pMDH8.png" width="500">
</p>

### Window Function

The fact that the impulse response does not decay to 0 make this FIR filter impractical. The common solution is to apply a [window function](https://en.wikipedia.org/wiki/Window_function). Scilab provides the implementation for several different [window functions](https://help.scilab.org/docs/6.0.2/en_US/window.html). The use of window function on the impulse response of an ideal low pass filter will have the following outcome.
```matlab
h_l = h_l .* window('kr', length(h_l), 8)
```
<p align="center">
<img src="https://i.imgur.com/OB3otgw.png" width="500">
</p>

### Evaluating the result

As a result of applying the window function, the frequency response of the filter changed. One can visually evaluate the frequency response by looking at its DFT transform
```matlab
plot2d("nl", frequencies, abs(fft(h_l)), color("blue"))
```
<p align="center">
<img src="https://i.imgur.com/fcNU4xM.png" width="500">
</p>

The quality of the filter is further decided by how well it is able to suppress frequencies in the stopband. The worst suppression gain in the stopband for the current filter is between $$10^{-5}$$ and $$10^{-4}$$. If the result is not satisfactory, one can try different windows, different initial frequency response, or longer FIR filter. If there is no good solution, another design method should be applied. 


### Applying the filter

One can apply the filter simply by performing the convolution between the signal and the designed filter response. The spectrogram investigation shows that the filter works as expected.

<p align="center">
<img src="https://i.imgur.com/BBRkWmJ.png" width="500">
</p>

The recording after filtering: [audio](https://www.dropbox.com/s/s3tojy5zv001fsz/signal_filtered.wav?dl=1).


## Canceling undesired effects

Consider some signal $$s$$ passing through an environment $$h$$. This environment has a certain effect on the signal. Denote the signal affected by the environment as $$\tilde{s}$$. 

<p align="center">
<img src="https://i.imgur.com/0dOuYXC.png" width="150">
</p>

The output of the system characterized by $$h$$ can be written as 
$$
\tilde{s} = h * s
$$
where $$*$$ represents convolution. This is essentially the way an echo is created. The audio signal $$s$$ propagates through the room to the signal receiver (microphone) along many paths. This multi-path propagation causes the echo to appear. 

<p align="center">
<img src="https://i.imgur.com/WQ5sVtn.png" width="200">
</p>

In this scenario, the *effect of the room* is undesired. This effect can be canceled by an additional filtering step so that 

$$
\begin{aligned}
s &= \tilde{h} * h * s \\
&= \delta * s \\
\end{aligned}
$$

As we know, the convolution of any sequence with a Kronecker sequence does not change the original sequence. 

The design of the filter $$\tilde{h}$$ is not straightforward in the time domain but is much easier in the frequency domain. 

Assume that the impulse response $$h$$ is a discrete-time sequence of length $$N$$ (shown on the figure below)

<p align="center">
<img src="https://i.imgur.com/zJf0V49.png" width="500">
</p>


DFT for this waveform can be calculated as 
$$
F_k = \sum_{n=0}^{N-1} s_n\cdot e^{-\frac{i 2\pi}{N}kn}
$$
where $$F_k, k=0..N-1$$ is the $$k^{th}$$ component in DFT spectrum. Each of these components can be represented in the form 
$$
F_k = A_k \cdot e^{-i \phi_k}
$$
where $$A_k =  \vert F_k \vert $$ and $$\phi_k = \arctan\frac{Imag(F_k)}{Real(F_k)}$$. We can characterize the signal by looking at the amplitude responce $$(A_k)$$ and phase responce $$(\phi_k)$$.

<p align="center">
<img src="https://i.imgur.com/OUjcs9D.png" width="250">
<img src="https://i.imgur.com/CTY0HxA.png" width="250">
</p>

The first figure shows the amplitude of the frequency components of $$h$$. Note that the image is nearly mirrored for the components with $$k>100000$$ (second half of the spectrum). The two parts of the spectrum are dependant, and this is a consequence of the Shannon-Nyquist (Kotelnikov) sampling theorem. 

The second figure shows the phase response of the system defined by $$h$$. Phase response shows how much phase delay the system introduces for the particular component of the spectrum. The symmetry of the spectrum comes from the fact that real-valued sequences are Hermitian symmetric in the frequency domain.

Amplitude and phase response are the two characteristics that completely describe the Linear Shift Invariant (LSI) system (and any LSI filter).

The filter $$\tilde{h}$$ that cancels the effect of $$h$$ should revert the changes in amplitude and phase introduced by $$h$$. We can find such $$\tilde{h}$$ by solving 
$$
\delta = \tilde{h} * h \iff \Delta(e^{-j\omega}) = \tilde{H}(e^{-j\omega}) H(e^{-j\omega})
$$

To understand how the solution for $$\tilde{H}(e^{-j\omega})$$ would look, inspect the DFT spectrum of a Kronecker sequence

<p align="center">
<img src="https://i.imgur.com/qL8ckbu.png" width="500">
</p>

<p align="center">
<img src="https://i.imgur.com/mRhmeES.png" width="250">
<img src="https://i.imgur.com/hNFNa4V.png" width="250">
</p>

From these figures you can see that 
$$
\Delta(e^{-j\omega})_k = 1 e^{-i \omega 0}
$$
for all $$k$$. Hense, the solution for $$\tilde{h}$$ can be found from 
$$
\tilde{H}(e^{-j\omega}) = \frac{H^*(e^{-j\omega})}{ \vert H(e^{-j\omega}) \vert ^2}
$$

Note that this solution exists only if all the components of $$H(e^{-j\omega})$$ are different from 0. For the filter above, the amplitude and phase spectrums are 

<p align="center">
<img src="https://i.imgur.com/uW0NJg9.png" width="250">
<img src="https://i.imgur.com/U8oW4r5.png" width="250">
</p>

Inverse transform reveals similar issue that was observed for the ideal low-pass filter
<p align="center">
<img src="https://i.imgur.com/a5pjntI.png" width="500">
</p>

After adding linear phase and window function, one will obtain a similar impulse response
<p align="center">
<img src="https://i.imgur.com/wDAl9UD.png" width="500">
</p>

The result of canceling the effect of echo can be observed on audio samples:
- [with echo](https://www.dropbox.com/s/x2oqc1t8m0706qj/speech_echo.wav?dl=1)
- [echo canceled](https://www.dropbox.com/s/byqf7s2kcwpjb2a/speech_echo_canceled_trim.wav?dl=1)

One can hear residual effects that appear after canceling echo. This comes from the fact that the impulse characteristic of the inverse filter appears to be very long at best and infinite at worst. After applying the window function, the filter is no longer capable of perfectly restoring the original signal.
