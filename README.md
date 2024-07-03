# noise-reduction
noise reduction
This model is Mossformer2, originally a speaker separation model. During the training process, it was found that the model has a strong noise reduction effect while separating the speakers. Now, the noise reduction model has been separated.It is recommended to use GPU for inference. After testing, the inference speed of GPU is dozens of times faster than that of CPU.
Only supports 1 channel audio with a sampling rate of 16000

这个模型是mossformer2，原来是一个说话人分离模型，在训练过程中，发现模型在说话人分离的同时具备很强的降噪效果，现在把降噪模型分离出来。建议使用GPU进行推理，经过测试，GPU的推理速度是CPU的数十倍。
只支持音频单通道，采样率为16000
