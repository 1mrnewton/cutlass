/*
 * The libavformat side of `cargo bench -p cutlass-caf`: the same three
 * jobs the Rust side does, in one process, `repeat` times.
 *
 *     ffmpeg_caf <open|read|seek> <file.caf> <repeat> <range>
 *
 * open: avformat_open_input with the CAF demuxer named, so no format
 *       probe runs (CafDemux::open does not probe either), and close. It
 *       reads the header and the packet table, as CafDemux::open does;
 *       avformat_find_stream_info is not called, since it reads and decodes
 *       packets.
 * read: open, then av_read_frame until the end; every packet is dropped.
 * seek: open once, then `repeat` seeks to frame times in [0, range) from
 *       the same xorshift sequence the Rust side uses, each followed by one
 *       packet read. Times are shifted by the first packet's time, since
 *       libavformat starts CAF packets at minus the priming.
 *
 * It prints what it read so the two sides can be compared. Built by the
 * bench with the system C compiler against the FFmpeg pkg-config finds.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavformat/avformat.h>

static uint64_t state = 0x5EEDCAFEULL;

static uint64_t next(void) {
    uint64_t x = state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static int fail(const char *what, int code) {
    char text[128];
    av_strerror(code, text, sizeof text);
    fprintf(stderr, "ffmpeg_caf: %s: %s\n", what, text);
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: ffmpeg_caf <open|read|seek> <file.caf> <repeat> <range>\n");
        return 1;
    }
    const char *mode = argv[1];
    const char *path = argv[2];
    long long repeat = atoll(argv[3]);
    long long range = atoll(argv[4]);
    av_log_set_level(AV_LOG_ERROR);
    const AVInputFormat *caf = av_find_input_format("caf");
    if (!caf) {
        fprintf(stderr, "ffmpeg_caf: this FFmpeg has no CAF demuxer\n");
        return 1;
    }
    AVPacket *packet = av_packet_alloc();
    long long packets = 0, bytes = 0, check = 0;
    if (strcmp(mode, "open") == 0 || strcmp(mode, "read") == 0) {
        int read = strcmp(mode, "read") == 0;
        for (long long turn = 0; turn < repeat; turn++) {
            AVFormatContext *format = NULL;
            int code = avformat_open_input(&format, path, caf, NULL);
            if (code < 0) return fail("open", code);
            packets = 0;
            bytes = 0;
            while (read && (code = av_read_frame(format, packet)) >= 0) {
                packets++;
                bytes += packet->size;
                check += packet->data[0];
                av_packet_unref(packet);
            }
            if (read && code != AVERROR_EOF) return fail("read", code);
            avformat_close_input(&format);
        }
    } else if (strcmp(mode, "seek") == 0) {
        AVFormatContext *format = NULL;
        int code = avformat_open_input(&format, path, caf, NULL);
        if (code < 0) return fail("open", code);
        code = av_read_frame(format, packet);
        if (code < 0) return fail("first packet", code);
        int64_t first = packet->pts;
        av_packet_unref(packet);
        for (long long turn = 0; turn < repeat; turn++) {
            int64_t target = first + (int64_t)(next() % (uint64_t)range);
            code = av_seek_frame(format, 0, target, AVSEEK_FLAG_BACKWARD);
            if (code < 0) return fail("seek", code);
            code = av_read_frame(format, packet);
            if (code < 0) return fail("read after seek", code);
            packets++;
            bytes += packet->size;
            check += packet->pts - first;
            av_packet_unref(packet);
        }
        avformat_close_input(&format);
    } else {
        fprintf(stderr, "ffmpeg_caf: no mode %s\n", mode);
        return 1;
    }
    av_packet_free(&packet);
    printf("packets %lld bytes %lld check %lld\n", packets, bytes, check);
    return 0;
}
