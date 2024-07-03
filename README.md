# noise-reduction
noise reduction
This model is Mossformer2, originally a speaker separation model. During the training process, it was found that the model has a strong noise reduction effect while separating the speakers. Now, the noise reduction model has been separated.It is recommended to use GPU for inference. After testing, the inference speed of GPU is dozens of times faster than that of CPU.

这个模型是mossformer2，原来是一个说话人分离模型，在训练过程中，发现模型在说话人分离的同时具备很强的降噪效果，现在把降噪模型分离出来。建议使用GPU进行推理，经过测试，GPU的推理速度是CPU的数十倍。


import onnx
import onnxruntime as ort
import numpy as np
import soundfile as sf

def save_result(est_source):
    signal = est_source[0, :, 0]
    signal = signal / np.abs(signal).max() * 0.5
    signal = signal[np.newaxis, :]
    output = (signal * 32768).astype(np.int16).tobytes()
    save_file = f'output_spk0.wav'
    sf.write(save_file, np.frombuffer(output, dtype=np.int16), 16000)

onnx_model_path = 'simple_model.onnx'
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
ort_session = ort.InferenceSession(onnx_model_path)
input_data,sr = sf.read('output_16000.wav')
if sr!=16000:raise 'Only supports 16000 Hz'
if input_data.ndim>1:raise 'Only supports 1 channel'
input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
input_name = ort_session.get_inputs()[0].name
outputs = ort_session.run(None, {input_name: input_data})
output_data = outputs[0]
save_result(output_data)
