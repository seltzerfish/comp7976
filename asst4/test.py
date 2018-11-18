def similar(a, b):
    max_count = 0
    for i in range(len(a)):
        count = 0
        b_pointer = 0
        for x in range(i, len(a)):
            if b_pointer >= len(b):
                break
            for j in range(b_pointer, len(b)):
                q, r = a[x], b[j]
                if a[x] == b[j]:
                    b_pointer = j + 1
                    count += 1
                    break
        max_count = max(count, max_count)
    return max_count

# print(similar("san", "francisco"))
# print(similar("Ala", "bama"))
# print(similar("abba", "bba"))


def decode_variants(inp):
    if not inp:
        return 1
    if len(inp) <= 1:
        if 0 < int(inp) <= 9:
            return 1
        else:
            return 0
    one_char = int(inp[0])
    two_char = int(inp[0:2])
    if 10 <= two_char <= 26:
        return decode_variants(inp[1:]) + decode_variants(inp[2:])
    else:
        return decode_variants(inp[1:])

print(decode_variants("3"))

print(decode_variants("12"))