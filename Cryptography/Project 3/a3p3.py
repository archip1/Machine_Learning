def solve3(m, sequence, a, b, c):
    r2, r3, r4, r5, r6 = tuple(sequence)
    # r6 = a*r5 + b*r4 + c
    # r5 = a*r4 + b*r3 + c
    # r4 = a*r3 + b*r2 + c

    # r3 = a*r2 + b*r1 + c
    # b*r1 = r3 - a*r2 - c 
    p = r3 - a*r2 - c
    b_inv = pow(b, -1, m)
    r1 = b_inv * p
    r1 %= m

    # r2 = a*r1 + b*r0 + c
    # b*r0 = r2 - a*r1 - c 
    q = r2 - a*r1 - c 
    b_inv = pow(b, -1, m)
    r0 = b_inv * q
    r0 %= m

    # r7 = a*r6 + b*r5 + c
    r7 = a*r6 + b*r5 + c
    r7 %= m

    return [r0, r1, r7]

if __name__ == "__main__" :
    print(solve3(467, [28, 137, 41, 118, 105], 37, 59, 325))