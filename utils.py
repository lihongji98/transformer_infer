from typing import List

import numpy as np


def softmax(x, dim=-1):
    x_max = np.max(x, axis=dim, keepdims=True)
    x_shifted = x - x_max

    x_exp = np.exp(x_shifted)
    x_sum = np.sum(x_exp, axis=dim, keepdims=True)

    softmax_values = x_exp / x_sum

    return softmax_values


def beam_search(session, encoder_input, decoder_input, beam_width=5, end_token=2, max_length=128):
    beam = [(decoder_input.reshape(-1), 0)]

    encoder_input_name = session.get_inputs()[0].name
    decoder_input_name = session.get_inputs()[1].name
    output_name = session.get_outputs()[0].name

    for _ in range(max_length):
        new_beam = []
        for sequence, score in beam:
            if sequence.reshape(-1)[-1] == end_token:
                new_beam.append((sequence, score))
                continue

            decoder_input = sequence.reshape(1, -1)
            output = session.run([output_name], {encoder_input_name: encoder_input, decoder_input_name: decoder_input})
            next_token_probs = softmax(output[0], dim=-1)[:, -1, :].flatten()
            top_tokens = np.argsort(next_token_probs)[-beam_width:]
            for token in top_tokens:
                new_sequence = np.concatenate((sequence, [token]), axis=0)
                new_score = score + np.log(next_token_probs[token])
                new_beam.append((new_sequence, new_score))

        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_width]

    return beam[0][0]


def glue_tokens_sentence(output: List[str]) -> str:
    return_sentence = ""
    for token in output:
        if token[-2:] != "@@":
            return_sentence += token + " "
        else:
            return_sentence += token[:-2]

    return return_sentence.rstrip()
