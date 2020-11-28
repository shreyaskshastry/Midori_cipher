from math import floor

import numpy as np

from constants import roundConstants
from sboxes import Sb1


arguments = []
outputs = []


def SSbi(x, i):
    """
    SSbi implementation function
    """
    # x0 is the most significant bit, x7 the least significant
    x0 = (x >> 7) & 1
    x1 = (x >> 6) & 1
    x2 = (x >> 5) & 1
    x3 = (x >> 4) & 1
    x4 = (x >> 3) & 1
    x5 = (x >> 2) & 1
    x6 = (x >> 1) & 1
    x7 = x & 1
    binResult = []
    if i == 0:
        n0 = Sb1([x4, x1, x6, x3])
        n1 = Sb1([x0, x5, x2, x7])
        binResult = [n1[0], n0[1], n1[2], n0[3],
                     n0[0], n1[1], n0[2], n1[3]]
    elif i == 1:
        n0 = Sb1([x1, x6, x7, x0])
        n1 = Sb1([x5, x2, x3, x4])
        binResult = [n0[3], n0[0], n1[1], n1[2],
                     n1[3], n1[0], n0[1], n0[2]]
    elif i == 2:
        n0 = Sb1([x2, x3, x4, x1])
        n1 = Sb1([x6, x7, x0, x5])
        binResult = [n1[2], n0[3], n0[0], n0[1],
                     n0[2], n1[3], n1[0], n1[1]]
    elif i == 3:
        n0 = Sb1([x7, x4, x1, x2])
        n1 = Sb1([x3, x0, x5, x6])
        binResult = [n1[1], n0[2], n0[3], n1[0],
                     n0[1], n1[2], n1[3], n0[0]]
    result = int("".join(binResult), 2)
    return result


def SubCell(state):
    """
     As described in paper
     si =  SSb(i mod 4)[si] where 0 <= i <= 15.
     Is applied here
    """
    for col in range(4):
        for row in range(4):
            i = col * 4 + row
            si = state[row, col]
            permuted = SSbi(si, i % 4)
            state[row, col] = permuted
            arguments.append(si)
            outputs.append(permuted)
    return state


def ShuffleCell(state):
    """
    As described in the paper
    (s0, s1,..., s15) <=  (s0, s10, s5, s15, s14, s4, s11, s1, s9, s3, s12, s6, s7, s13, s2, s8)
    permutation is applied here
    """
    sDict = {}
    for col in range(4):
        for row in range(4):
            i = col * 4 + row
            sDict['s' + str(i)] = state[row, col]
    newState = np.matrix([
        [sDict['s0'], sDict['s14'], sDict['s9'], sDict['s7']],
        [sDict['s10'], sDict['s4'], sDict['s3'], sDict['s13']],
        [sDict['s5'], sDict['s11'], sDict['s12'], sDict['s2']],
        [sDict['s15'], sDict['s1'], sDict['s6'], sDict['s8']]
    ])
    return newState


def MixColumn(state):
    """
    As described in the paper
    (si; si+1; si+2; si+3)^T =  M * (si; si+1; si+2; si+3)^T and i = 0; 4; 8; 12.
    Is applied here.
    Note that the calculations are done in GF(2^8).
    """
    m = np.matrix([
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0]
    ]
    )
    for col in range(4):
        stateCol = np.squeeze(np.asarray(state[:, col]))
        result = UpdateColumn(m, stateCol)
        state[:, col] = UpdateColumn(m, stateCol)
    return state


def UpdateColumn(m, col):
    results = np.zeros(4, dtype=np.uint8)
    for row in range(4):
        matrixRow = np.squeeze(np.asarray(m[row, :]))
        for (a, b) in zip(matrixRow, col):
            results[row] ^= PolyMult(a, b)
    newColumn = np.transpose(np.asmatrix(results))
    return newColumn


def PolyMult(p1, p2, debug=False):
    """
    Multiply two numbers in the GF(2^8) finite field defined
    See http://stackoverflow.com/questions/13202758/multiplying-two-polynomials
    For info 
    """
    binP2 = bin(p2)[2:].zfill(8)
    mult = 0
    if p1 == 0 or p2 == 0:
        return 0
    for i in range(8):
        bit = binP2[i]
        if bit == "1":
            mult ^= (p1 << (7 - i))
    reducPoly = int("100011011", 2)
    while True:
        if GetMSBIndex(mult) < GetMSBIndex(reducPoly):
            break
        elif GetMSBIndex(mult) == GetMSBIndex(reducPoly):
            mult ^= reducPoly
        else:
            degreeDiff = GetMSBIndex(mult) - GetMSBIndex(reducPoly)
            mult ^= (reducPoly << degreeDiff)
    return mult


def GetMSBIndex(n):
    """
     Getting most significiant bit
    """
    ndx = 0
    while 1 < n:
        n = (n >> 1)
        ndx += 1
    return ndx


def GetLSBIndex(n):
    """
    Getting lest significiant bit
    """
    return GetMSBIndex(n & -n)


def SplitByN(seq, n):
    """
    Split function
    """
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def StateToBinary(state):
    """
    Binary String converter
    """
    binary = ""
    for col in range(4):
        for row in range(4):
            si = state[row, col]
            binary += bin(si)[2:].zfill(8)
    return binary


def RoundKeyGen(key):
    """
    As described in paper round key generation
    """
    print("The original key is:", hex(int(key, 2))[2:].zfill(32))
    roundKeys = []
    keyBytes = SplitByN(key, 8)
    for r in range(19):
        rConst = roundConstants[r]
        newRoundKeyBytes = []
        for i in range(16):
            col = floor(i / 4)
            row = i % 4
            bit = rConst[row, col]
            curRoundKeyByte = keyBytes[i]
            newRoundKeyByte = bin(int(curRoundKeyByte, 2) ^ bit)[2:].zfill(8)
            newRoundKeyBytes.append(newRoundKeyByte)
        roundKey = ''.join(newRoundKeyBytes)
        roundKeys.append(roundKey)
    return roundKeys


def KeyAdd(state, roundKeyI):
    """
    Key addition
    """
    roundKeyBytesArray = SplitByN(roundKeyI, 8)
    for col in range(4):
        for row in range(4):
            si = state[row, col]
            roundKeyByte = int(roundKeyBytesArray[col * 4 + row], 2)
            state[row, col] = roundKeyByte ^ si
    return state


def InitializeState(binaryString):
    """
    State initializer
    """
    state = np.zeros(shape=(4, 4), dtype=np.uint8)
    plaintextBytes = SplitByN(binaryString, 8)
    for col in range(4):
        for row in range(4):
            binary = plaintextBytes[col * 4 + row]
            state[row, col] = int(binary, 2)
    return np.matrix(state)


def InvShuffleCell(state):
    """
    As described in paper InvShuffel for decryption
    """
    sDict = {}
    for col in range(4):
        for row in range(4):
            i = col * 4 + row
            sDict['s' + str(i)] = state[row, col]
    newState = np.matrix([
        [sDict['s0'], sDict['s5'], sDict['s15'], sDict['s10']],
        [sDict['s7'], sDict['s2'], sDict['s8'], sDict['s13']],
        [sDict['s14'], sDict['s11'], sDict['s1'], sDict['s4']],
        [sDict['s9'], sDict['s12'], sDict['s6'], sDict['s3']]
    ])
    return newState


def RoundConstantsToBin(encryption=True):
    """
    As described in the paper Round constant generator
    """
    print("Round constants to bin")
    binRKs = []
    for RK in roundConstants:
        binRK = ""
        for i in range(16):
            col = floor(i / 4)
            row = i % 4
            bit = RK[row, col]
            byte = bin(bit)[2:].zfill(8)
            binRK += byte
        if not encryption:
            binRK = LinearInverse(binRK)
        binRKs.append(binRK)
        print(binRK)


def LinearInverse(roundkey):
    """
    L^-1 which does the permutation as
    (s0, s1, ..., s15) <=  (s0, s7, s14, s9, s5, s2, s11, s12, s15, s8, s1, s6, s10, s13, s4, s3):
    """
    state = InitializeState(roundkey)
    newstate = InvShuffleCell(MixColumn(state))
    return StateToBinary(newstate)


def MidoriEncrypt(plaintext, key, r):
    """
    Encryption
    """
    plaintext = bin(int(plaintext, 16))
    key = bin(int(key, 16))
    plaintext = plaintext[2:].zfill(128)
    key = key[2:].zfill(128)
    print("Plaintext", plaintext)
    print("key", key)
    state = InitializeState(plaintext)
    state = KeyAdd(state, key)
    RKs = RoundKeyGen(key)
    for i in range(r - 1):
        state = SubCell(state)
        state = ShuffleCell(state)
        state = MixColumn(state)
        state = KeyAdd(state, RKs[i])
    state = SubCell(state)
    y = KeyAdd(state, key)
    ciphertext = int(StateToBinary(y), 2)
    print("The ciphertext is as follows:\n0x{0:02x}".format(ciphertext))
    return "0x{0:02x}".format(ciphertext)


def MidoriDecrypt(ciphertext, key, r):
    """
    Decryption
    """
    ciphertext = bin(int(ciphertext, 16))
    key = bin(int(key, 16))
    ciphertext = ciphertext[2:].zfill(128)
    key = key[2:].zfill(128)
    print("Ciphertext:", ciphertext)
    state = InitializeState(ciphertext)
    state = KeyAdd(state, key)
    RKs = RoundKeyGen(key)
    RKs = list(reversed(RKs))
    for i in range(r - 1):
        state = SubCell(state)
        state = MixColumn(state)
        state = InvShuffleCell(state)
        state = KeyAdd(state, LinearInverse(RKs[i]))
    state = SubCell(state)
    x = KeyAdd(state, key)
    plaintext = int(StateToBinary(x), 2)
    print("The plaintext is as follows:\n0x{0:02x}\n".format(plaintext))
    return "0x{0:02x}".format(plaintext)


def generate_sbox_lookup_tables():
    """
    Generating the S-box
    see https://stackoverflow.com/a/12638477 
    For original resource
    """
    desired_width = 320
    np.set_printoptions(linewidth=desired_width)
    bit_length = 8
    sbox_width = 2 ** (bit_length // 2)
    sbox_lookup_tables = []
    for i in range(4):
        sbox_lookup_table = np.zeros((sbox_width, sbox_width), dtype=np.int32)
        for val in range(np.iinfo(np.uint8).max + 1):
            least_significant_nibble = val & 0xF
            most_significant_nibble = (val >> 4) & 0xF
            s_box_val = SSbi(val, i)
            sbox_lookup_table[most_significant_nibble,
                              least_significant_nibble] = s_box_val
        print("Sbox table values for SSb{}".format(i))
        for row in range(sbox_lookup_table[0].shape):
            row_values = sbox_lookup_table[row]
            row_values = ["{0:#0{1}x}".format(val, 4) for val in row_values]
            print(row_values)
        print("\n")

        sbox_lookup_tables.append(sbox_lookup_table)
    return sbox_lookup_tables


def main():
    """
    Main function
    """
    plaintext = "0x51084ce6e73a5ca2ec87d7babc297543"
    key = "0x687ded3b3c85b3f35b1009863e2a8cbf"
    r = 20  # Number of rounds
    c = MidoriEncrypt(plaintext, key, r)
    p = MidoriDecrypt(c, key, r)
    if p.lower() == plaintext.lower() or int(p, 16) == int(plaintext, 16):
        print("Encryption and decryption working as expected.")
        print("Encryption of {} lead to the ciphertext: {}".format(plaintext, c))
        print("Decryption of {} lead to the plaintext: {}".format(c, p))
    else:
        print("Encryption and decryption not working as expected.")
        print("Encryption of {} lead to the ciphertext: {}".format(plaintext, c))
        print("Decryption of {} lead to the plaintext: {}".format(c, p))


if __name__ == "__main__":
    main()
