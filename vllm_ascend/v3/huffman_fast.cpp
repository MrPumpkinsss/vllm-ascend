#include <cstddef>
#include <cstdint>

static inline uint32_t peek_bits(const uint8_t* data, int64_t bit_pos, int width) {
    uint64_t acc = 0;
    const int64_t byte = bit_pos >> 3;
    const int r = static_cast<int>(bit_pos & 7);
    for (int k = 0; k < 8; ++k) acc = (acc << 8) | static_cast<uint64_t>(data[byte + k]);
    return static_cast<uint32_t>((acc >> (64 - r - width)) & ((1ULL << width) - 1));
}

static inline void put_bits(uint8_t* out, int64_t bit_pos, uint32_t value, int width) {
    for (int i = width - 1; i >= 0; --i) {
        const int64_t p = bit_pos + (width - 1 - i);
        if ((value >> i) & 1u) out[p >> 3] |= static_cast<uint8_t>(1u << (7 - (p & 7)));
    }
}

extern "C" int64_t huff_encode(
    const uint32_t* code_values,
    const uint8_t* code_lengths,
    const int64_t* ctx,
    const uint16_t* symbols,
    const uint16_t* exact_codes,
    int64_t n,
    uint8_t* out,
    int64_t capacity_bits,
    uint16_t escape_symbol
) {
    int64_t pos = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t c = ctx[i];
        const uint16_t sym = symbols[i];
        const int len = code_lengths[(c << 10) + sym];
        if (len <= 0 || len > 32) return -1;
        if (pos + len > capacity_bits) return -2;
        put_bits(out, pos, code_values[(c << 10) + sym], len);
        pos += len;
        if (sym == escape_symbol) {
            if (pos + 16 > capacity_bits) return -2;
            put_bits(out, pos, exact_codes[i], 16);
            pos += 16;
        }
    }
    return pos;
}

extern "C" int64_t huff_decode(
    const uint32_t* lut,
    const int64_t* long_offsets,
    const uint64_t* long_starts,
    const uint64_t* long_ends,
    const uint32_t* long_values,
    const int64_t* ctx,
    int64_t n,
    const uint8_t* data,
    int64_t total_bits,
    uint16_t* decoded,
    uint16_t* escapes,
    int32_t long_width,
    uint16_t escape_symbol
) {
    constexpr uint32_t kInvalid = 0xFFFFFFFFu;
    int64_t pos = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (pos >= total_bits) return -1;
        const int64_t c = ctx[i];
        const uint32_t window = peek_bits(data, pos, 16);
        uint32_t packed = lut[(c << 16) + window];
        if (packed == kInvalid) {
            const uint32_t key = peek_bits(data, pos, long_width);
            const int64_t lo = long_offsets[c];
            const int64_t hi = long_offsets[c + 1];
            int64_t a = lo, b = hi;
            while (a < b) {
                const int64_t mid = a + ((b - a) >> 1);
                if (long_starts[mid] <= key) a = mid + 1;
                else b = mid;
            }
            --a;
            if (a < lo || key >= long_ends[a]) return -2;
            packed = long_values[a];
        }
        const int64_t length = static_cast<int64_t>(packed >> 20);
        const uint32_t sym = packed & ((1u << 20) - 1);
        decoded[i] = static_cast<uint16_t>(sym);
        pos += length;
        if (sym == escape_symbol) {
            if (pos + 16 > total_bits) return -3;
            escapes[i] = static_cast<uint16_t>(peek_bits(data, pos, 16));
            pos += 16;
        }
    }
    return pos;
}
