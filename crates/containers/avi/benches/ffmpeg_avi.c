/*
 * The FFmpeg side of the AVI benchmark: the same work `examples/bench.rs`
 * and `examples/remux.rs` do, done with libavformat. A test oracle only;
 * nothing in Cutlass links it.
 *
 *     ffmpeg_avi open   <file> <repeat>   open the file and close it
 *     ffmpeg_avi read   <file> <repeat>   open and read every packet
 *     ffmpeg_avi seek   <file> <seeks>    open, then seek stream 0 and read one packet, <seeks> times
 *     ffmpeg_avi remux  <in> <out>        copy every packet into a new AVI
 *
 * `avformat_find_stream_info` is not called by `open`, `read` or `seek`:
 * AVI names every stream's codec and size in its header, and that call
 * decodes frames, which the Rust side does not do. `remux` calls it, because
 * FFmpeg's AVI muxer needs the audio frame size it finds. libavformat still runs the parsers the AVI demuxer asks for
 * on some streams (MPEG audio, for one) inside `av_read_frame`.
 *
 * Build with the system compiler against the FFmpeg `pkg-config` finds:
 *
 *     cc -O2 -o ffmpeg_avi ffmpeg_avi.c $(pkg-config --cflags --libs libavformat libavcodec libavutil)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavformat/avformat.h>

static int fail(const char *what, int code) {
    char text[128];
    av_strerror(code, text, sizeof text);
    fprintf(stderr, "ffmpeg_avi: %s: %s\n", what, text);
    return 1;
}

static int open_file(const char *path, AVFormatContext **file) {
    *file = NULL;
    int code = avformat_open_input(file, path, NULL, NULL);
    return code < 0 ? fail("open", code) : 0;
}

static int read_all(const char *path, long long *count, long long *bytes) {
    AVFormatContext *file;
    if (open_file(path, &file)) return 1;
    AVPacket *packet = av_packet_alloc();
    unsigned seen = 0;
    for (;;) {
        int code = av_read_frame(file, packet);
        if (code == AVERROR_EOF) break;
        if (code < 0) return fail("read", code);
        *count += 1;
        *bytes += packet->size;
        if (packet->size) seen += packet->data[0];
        av_packet_unref(packet);
    }
    av_packet_free(&packet);
    avformat_close_input(&file);
    return seen == 0xffffffffu ? 2 : 0;
}

/* The same spread of targets as the Rust side: a fixed linear
 * congruential walk over the stream's length. */
static int seek(const char *path, int seeks) {
    AVFormatContext *file;
    if (open_file(path, &file)) return 1;
    AVStream *stream = file->streams[0];
    long long length = stream->duration;
    if (length <= 0) {
        fprintf(stderr, "ffmpeg_avi: stream 0 has no length\n");
        return 1;
    }
    AVPacket *packet = av_packet_alloc();
    unsigned long long state = 1;
    long long landed = 0;
    for (int turn = 0; turn < seeks; turn++) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        long long target = (long long)((state >> 33) % (unsigned long long)length);
        int code = av_seek_frame(file, 0, target, AVSEEK_FLAG_BACKWARD);
        if (code < 0) return fail("seek", code);
        code = av_read_frame(file, packet);
        if (code < 0) return fail("read after seek", code);
        landed += packet->size;
        av_packet_unref(packet);
    }
    av_packet_free(&packet);
    avformat_close_input(&file);
    printf("seeks %d bytes %lld\n", seeks, landed);
    return 0;
}

static int remux(const char *from, const char *to) {
    AVFormatContext *input;
    if (open_file(from, &input)) return 1;
    /* The AVI muxer picks each audio stream's time base from its frame
     * size, which only this call fills in; without it AAC times collide. */
    int code = avformat_find_stream_info(input, NULL);
    if (code < 0) return fail("stream info", code);
    AVFormatContext *output = NULL;
    code = avformat_alloc_output_context2(&output, NULL, "avi", to);
    if (code < 0) return fail("output", code);
    for (unsigned i = 0; i < input->nb_streams; i++) {
        AVStream *stream = avformat_new_stream(output, NULL);
        if (!stream) return fail("stream", AVERROR(ENOMEM));
        code = avcodec_parameters_copy(stream->codecpar, input->streams[i]->codecpar);
        if (code < 0) return fail("parameters", code);
        stream->codecpar->codec_tag = input->streams[i]->codecpar->codec_tag;
        stream->time_base = input->streams[i]->time_base;
    }
    code = avio_open(&output->pb, to, AVIO_FLAG_WRITE);
    if (code < 0) return fail("create", code);
    code = avformat_write_header(output, NULL);
    if (code < 0) return fail("header", code);
    AVPacket *packet = av_packet_alloc();
    long long count = 0;
    for (;;) {
        code = av_read_frame(input, packet);
        if (code == AVERROR_EOF) break;
        if (code < 0) return fail("read", code);
        av_packet_rescale_ts(packet, input->streams[packet->stream_index]->time_base,
                             output->streams[packet->stream_index]->time_base);
        packet->pos = -1;
        code = av_write_frame(output, packet);
        if (code < 0) return fail("write", code);
        count += 1;
    }
    code = av_write_trailer(output);
    if (code < 0) return fail("trailer", code);
    av_packet_free(&packet);
    avio_closep(&output->pb);
    avformat_free_context(output);
    avformat_close_input(&input);
    printf("packets %lld\n", count);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: ffmpeg_avi <open|read|seek|remux> <file> <repeat|seeks|out>\n");
        return 1;
    }
    av_log_set_level(AV_LOG_ERROR);
    if (strcmp(argv[1], "remux") == 0) return remux(argv[2], argv[3]);
    int repeat = atoi(argv[3]);
    if (strcmp(argv[1], "seek") == 0) return seek(argv[2], repeat);
    if (strcmp(argv[1], "open") == 0) {
        for (int turn = 0; turn < repeat; turn++) {
            AVFormatContext *file;
            if (open_file(argv[2], &file)) return 1;
            avformat_close_input(&file);
        }
        printf("opens %d\n", repeat);
        return 0;
    }
    if (strcmp(argv[1], "read") == 0) {
        long long count = 0, bytes = 0;
        for (int turn = 0; turn < repeat; turn++) {
            count = 0;
            bytes = 0;
            if (read_all(argv[2], &count, &bytes)) return 1;
        }
        printf("packets %lld bytes %lld\n", count, bytes);
        return 0;
    }
    fprintf(stderr, "ffmpeg_avi: no mode %s\n", argv[1]);
    return 1;
}
