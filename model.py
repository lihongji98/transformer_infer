import re
from collections import defaultdict
from itertools import islice
import subprocess
from typing import List

import onnxruntime as rt
import numpy as np

from translator.utils import beam_search, glue_tokens_sentence


mosedecoder_path = "./translator/packages/mosesdecoder"
vocab_process_path = "./translator/voc"


class ONNXModelExecutor:
    def __init__(self, src: str, trg: str, voc_size=16000):
        self.src, self.trg = src, trg
        self.src_dict = self.load_voc_dict(src, voc_size)
        self.trg_dict = self.load_voc_dict(trg, voc_size)
        self.reverse_trg_dict = {v: k for k, v in self.trg_dict.items()}
        self.session = None
        self.line_symbols: List[str] = []

    @staticmethod
    def load_voc_dict(lang, voc_size):
        voc_table = defaultdict(lambda: len(voc_table))
        voc_table.default_factory = voc_table.__len__

        with open(f"{vocab_process_path}" + f"/voc_{lang}.txt", "r", encoding="utf-8") as voc_source:
            for line in islice(voc_source, voc_size):
                token, index_str = line.strip().split()
                voc_table[token] = int(index_str)

        return voc_table

    def load_onnx_model(self, model_path):
        try:
            self.session = rt.InferenceSession(model_path)
        except FileNotFoundError:
            raise RuntimeError("File is not found...")

    def preprocessing(self, lines_to_translate: str) -> str:
        command1 = ["perl", f"{mosedecoder_path}" + "/scripts/tokenizer/normalize-punctuation.perl", "-l", f"{self.src}"]
        command2 = ["perl", f"{mosedecoder_path}" + "/scripts/tokenizer/tokenizer.perl", "-l", f"{self.src}"]
        command3 = ["perl", f"{mosedecoder_path}" + "/scripts/recaser/truecase.perl", "--model",
                    f"{vocab_process_path}" + f"/truecase-model.{self.src}"]
        command4 = ["python", f"{vocab_process_path}" + "/apply_bpe.py", "-c", f"{vocab_process_path}" + f"/bpecode.{self.src}"]

        lines_to_translate = self.run_script(command1, lines_to_translate)
        lines_to_translate = self.run_script(command2, lines_to_translate)
        lines_to_translate = self.run_script(command3, lines_to_translate)
        lines_to_translate = self.run_script(command4, lines_to_translate)
        
        return lines_to_translate

    @staticmethod
    def run_script(cmd, lines_to_translate):
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True)
        lines_to_translate, errors = process.communicate(input=lines_to_translate)

        return lines_to_translate

    def convert_to_token_id(self, tokens: List[str]) -> List[int]:
        token_ids = [self.src_dict.get(tokens[i], self.src_dict["<unk>"]) for i in range(len(tokens))]
        token_ids = [1] + token_ids + [2]

        return token_ids

    def generate_encoder_decoder_input(self, lines_to_translate: str):
        lines_to_translate = self.preprocessing(lines_to_translate)
        lines_to_translate = re.findall(r'[^.!?]+[.!?]?', lines_to_translate)
        lines_to_translate = [s.strip() for s in lines_to_translate if s.strip()]

        for s in lines_to_translate:
            self.line_symbols.append(s[-1])

        tokens_ids = [self.convert_to_token_id(line_to_translate.split()) for line_to_translate in lines_to_translate]

        encoder_inputs, decoder_inputs = [], []
        for i in range(len(tokens_ids)):
            encoder_input = np.array(tokens_ids[i] + [0 for _ in range(128 - len(tokens_ids[i]))]).astype(np.int64).reshape(1, -1)
            decoder_input = np.array([1]).astype(np.int64).reshape(1, -1)
            encoder_inputs.append(encoder_input)
            decoder_inputs.append(decoder_input)

        return encoder_inputs, decoder_inputs

    def postprocessing(self, lines: str) -> str:
        command1 = ["perl", f"{mosedecoder_path}" + "/scripts/recaser/detruecase.perl"]
        command2 = ["perl", f"{mosedecoder_path}" + "/scripts/tokenizer/detokenizer.perl", "-l", f"{self.trg}"]
        lines = self.run_script(command1, lines)
        lines = self.run_script(command2, lines)

        return lines

    def infer(self, lines_to_translate: str):
        encoder_inputs, decoder_inputs = self.generate_encoder_decoder_input(lines_to_translate)

        outputs_buffer = ""
        for index, (encoder_input, decoder_input) in enumerate(zip(encoder_inputs, decoder_inputs)):
            output = beam_search(self.session, encoder_input, decoder_input, beam_width=1)
            output = [self.reverse_trg_dict.get(output[i], "<unk>") for i in range(len(output))][1:-1]
            output = glue_tokens_sentence(output)

            if not re.search(r'[.!?]$', output) and re.search(r'\w+$', output):
                output += " " + self.line_symbols[index]
            outputs_buffer += output + " "
        output = self.postprocessing(outputs_buffer)

        return output
