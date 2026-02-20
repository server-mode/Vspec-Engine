#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "vspec/quant/pack.h"

static int test_roundtrip(uint8_t bits) {
    const int8_t src[] = {-9, -8, -7, -4, -1, 0, 1, 2, 3, 4, 7, 8};
    const size_t count = sizeof(src) / sizeof(src[0]);

    const size_t bytes = vspec_quant_packed_bytes(count, bits);
    uint8_t* packed = (uint8_t*)malloc(bytes);
    int8_t* dst = (int8_t*)malloc(count);

    if (!packed || !dst) {
        free(packed);
        free(dst);
        return 0;
    }

    vspec_quant_pack_signed(src, count, bits, packed);
    vspec_quant_unpack_signed(packed, count, bits, dst);

    int ok = 1;
    for (size_t i = 0; i < count; ++i) {
        const int8_t expected = vspec_quant_clip_signed(src[i], bits);
        if (dst[i] != expected) {
            ok = 0;
            printf("bits=%u index=%zu expected=%d got=%d\n", (unsigned)bits, i, (int)expected, (int)dst[i]);
            break;
        }
    }

    free(packed);
    free(dst);
    return ok;
}

int main(void) {
    const uint8_t bits_list[] = {2, 3, 4};
    int all_ok = 1;

    for (size_t i = 0; i < sizeof(bits_list) / sizeof(bits_list[0]); ++i) {
        const uint8_t bits = bits_list[i];
        const int ok = test_roundtrip(bits);
        printf("quant roundtrip %u-bit: %s\n", (unsigned)bits, ok ? "PASS" : "FAIL");
        if (!ok) {
            all_ok = 0;
        }
    }

    return all_ok ? 0 : 1;
}
