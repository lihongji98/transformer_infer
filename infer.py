from collections import defaultdict
from itertools import islice
import subprocess
from typing import List

import onnxruntime as rt
import numpy as np


def softmax(x, dim=1):
    x_max = np.max(x, axis=dim, keepdims=True)
    x_shifted = x - x_max

    x_exp = np.exp(x_shifted)
    x_sum = np.sum(x_exp, axis=dim, keepdims=True)

    softmax_values = x_exp / x_sum

    return softmax_values


class ONNXModelExecutor:
    def __init__(self, src: str, trg: str, voc_size=16000):
        self.src, self.trg = src, trg
        self.src_dict = self.load_voc_dict(src, voc_size)
        self.trg_dict = self.load_voc_dict(trg, voc_size)
        self.reverse_trg_dict = {v: k for k, v in self.trg_dict.items()}
        self.session = None

    @staticmethod
    def load_voc_dict(lang, voc_size):
        voc_table = defaultdict(lambda: len(voc_table))
        voc_table.default_factory = voc_table.__len__

        with open(f"./voc/voc_{lang}.txt", "r", encoding="utf-8") as voc_source:
            for line in islice(voc_source, voc_size):
                token, index_str = line.strip().split()
                voc_table[token] = int(index_str)

        return voc_table

    def load_onnx_model(self, model_path):
        try:
            self.session = rt.InferenceSession(model_path)
        except FileNotFoundError:
            raise RuntimeError("File is not found...")

    def preprocessing(self, lines_to_translate: str) -> List[str]:
        command1 = ["perl", "/Users/lihongji/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl", "-l", "en"]
        command2 = ["perl", "/Users/lihongji/mosesdecoder/scripts/tokenizer/tokenizer.perl", "-l", f"{self.src}"]
        command3 = ["perl", "/Users/lihongji/mosesdecoder/scripts/recaser/truecase.perl", "--model", f"./voc/truecase-model.{self.src}"]
        command4 = ["python", "./voc/apply_bpe.py", "-c", "./voc/bpecode.en"]
        lines_to_translate = self.run_script(command1, lines_to_translate)
        lines_to_translate = self.run_script(command2, lines_to_translate)
        lines_to_translate = self.run_script(command3, lines_to_translate)
        lines_to_translate = self.run_script(command4, lines_to_translate)

        return lines_to_translate.split()

    @staticmethod
    def run_script(cmd, lines_to_translate):
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        lines_to_translate, errors = process.communicate(input=lines_to_translate)

        return lines_to_translate

    def convert_to_token_id(self, tokens: List[str]) -> List[int]:
        token_ids = [self.src_dict.get(tokens[i], self.src_dict["<unk>"]) for i in range(len(tokens))]
        token_ids = [1] + token_ids + [2]

        return token_ids

    def generate_encoder_decoder_input(self, lines_to_translate: str):
        preprocessed_lines = self.preprocessing(lines_to_translate)
        token_ids = self.convert_to_token_id(preprocessed_lines)
        encoder_input = np.array(token_ids + [0 for _ in range(128 - len(token_ids))]).astype(np.int64).reshape(1, -1)
        decoder_input = np.array([1]).astype(np.int64).reshape(1, -1)

        return encoder_input, decoder_input

    def postprocessing(self):
        pass

    def infer(self, lines_to_translate: str):
        encoder_input, decoder_input = self.generate_encoder_decoder_input(lines_to_translate)

        encoder_input_name = self.session.get_inputs()[0].name
        decoder_input_name = self.session.get_inputs()[1].name
        output_name = self.session.get_outputs()[0].name

        for _ in range(128):
            output = self.session.run([output_name],
                                 {encoder_input_name: encoder_input,
                                  decoder_input_name: decoder_input})

            top_token_id = softmax(output[0], dim=-1)[:, -1, :].flatten()
            top_token_id = np.argsort(top_token_id)[-1:]
            decoder_input = np.append(decoder_input, np.int64(top_token_id)).reshape(1, -1)

            if decoder_input[0][-1] == 2:
                break

        output = decoder_input[0]
        print(output)
        output = [self.reverse_trg_dict.get(output[i], "<unk>") for i in range(len(output))][1:-1]

        print(output)


if __name__ == "__main__":
    inferer = ONNXModelExecutor(src="no", trg="en")
    inferer.load_onnx_model("/Users/lihongji/PycharmProjects/gnn/No-En-Transformer.onnx")
    inferer.infer("god morgen, hvordan har du det i dag?")
