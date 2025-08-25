##### CARLOS CODE FILE ########
h, w = 123, 124
# compute how much to pad so h and w become multiples of 32
pad_h = (32 - h % 32) % 32

pad_w = (32 - w % 32) % 32

print(pad_h, pad_w)