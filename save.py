import pickle

# a="a"
# filename = "dump.txt"
# with open(filename, 'wb') as f:
#     pickle.dump(a, f)

filename="var.vc"
with open(filename, 'rb') as f:
    b = pickle.load(f)

print(b)