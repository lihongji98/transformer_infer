from model import ONNXModelExecutor


if __name__ == "__main__":
    sentence = "Ok, da har jeg reservert et dobbeltrom med to enkeltsenger med utsikt mot hagen fra fredag til søndag neste helg under navnet Hanne Nilsen. Høres dette greit ut?"
    inferer = ONNXModelExecutor(src="no", trg="en")
    inferer.load_onnx_model("No-En-Transformer.onnx")
    output = inferer.infer(sentence)
    print("raw sentence: ")
    print(sentence)
    print("")
    print("translated sentence: ") 
    print(output)