import vspec as torch

model = torch.load("sample_weights.vpt")
print("tensor_count=", model.tensor_count)
print(model.generate("hello"))
