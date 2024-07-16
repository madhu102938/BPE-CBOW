from tokenizer.BPE import MinBPE

obj = MinBPE()

all_text:str
with open('./data/Shakespeare.txt', 'r', encoding='utf-8') as f:
    all_text = f.read()

obj.train(all_text, 5000)

text = "hello, what is your name?"
print(obj.encode(text))
print(obj.display(text))