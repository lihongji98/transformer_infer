from model import ONNXModelExecutor


if __name__ == "__main__":
    sentence = "Nordmenn ... Litt en jevn innesluttet type. Men veldig hyggelig hvis du møter dem på tur i fjellet eller i skogen. Da sier vi alltid hei til hverandre. Men ikke hvis du møter hverandre på gata."
    inferer = ONNXModelExecutor(src="no", trg="en")
    inferer.load_onnx_model("No-En-Transformer.onnx")
    output = inferer.infer(sentence)
    print("raw sentence: ")
    print(sentence)
    print("")
    print("translated sentence: ") 
    print(output)