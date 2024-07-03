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
input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
input_name = ort_session.get_inputs()[0].name
outputs = ort_session.run(None, {input_name: input_data})
output_data = outputs[0]
print(output_data.shape)
save_result(output_data)