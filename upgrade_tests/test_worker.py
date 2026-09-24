import base64
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import wave

spec = importlib.util.spec_from_file_location('worker25', Path(__file__).resolve().parents[1] / 'handler.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)

class WorkerTests(unittest.TestCase):
    def test_controls_and_cleanup(self):
        paths = []
        def infer(**args):
            self.assertEqual(args['emo_text'], 'Subtly excited')
            self.assertTrue(args['use_emo_text'])
            self.assertEqual(args['duration_factor'], 1.11)
            paths.append(Path(args['spk_audio_prompt']))
            with wave.open(args['output_path'], 'wb') as audio:
                audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
                audio.writeframes(b'\x00\x01' * 240)
        with patch.object(w, 'get_model', return_value=Mock(infer=infer)):
            result = w.handler({'input': {'text': 'Hello!', 'ref_audio_b64': base64.b64encode(b'reference').decode(),
                                         'emotion': 'Subtly excited', 'duration_factor': 1.11}})
        self.assertTrue(result['ok'])
        self.assertEqual(result['model_version'], '2.5')
        self.assertEqual(result['sr'], 24000)
        self.assertFalse(paths[0].exists())

    def test_invalid_controls_never_load_model(self):
        with patch.object(w, 'get_model') as model:
            for extra in [{'duration_factor': float('nan')}, {'emotion_strength': 2}, {'lang': 'INVALID'}, {'emotion': []}]:
                result = w.handler({'input': {'text': 'Hello', **extra}})
                self.assertFalse(result['ok'])
            model.assert_not_called()

    def test_health_does_not_load_weights(self):
        with patch.object(w, 'get_model') as model:
            self.assertEqual(w.handler({'input': {'mode': 'health'}})['model_version'], '2.5')
            model.assert_not_called()
