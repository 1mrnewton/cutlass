/*
 * The libavformat side of the Matroska benchmark: the same three jobs the
 * `bench` example does with the Rust demuxer and muxer.
 *
 *     ffmpeg_matroska open  <file>
 *     ffmpeg_matroska read  <file>
 *     ffmpeg_matroska seek  <file> <count>
 *     ffmpeg_matroska remux <in> <out>
 *
 * `open` opens the file and reads no packet: the start-up and header cost
 * the other jobs include. `read` reads every packet of every stream and decodes none. `seek` seeks
 * the first stream `count` times, to the same spread of times the example
 * uses, backward to a keyframe at or before each, and reads one packet after
 * each. `remux` copies every packet into a new Matroska file (WebM when the
 * name ends in .webm) with libavformat's muxer, without a codec or a filter.
 *
 * `avformat_find_stream_info` is not called for `open`, `read` and `seek`:
 * it decodes the first frames to fill in stream parameters, which is codec
 * work the Rust side does not do, and the Matroska header already carries
 * what those jobs need. `remux` calls it, because libavformat's muxer wants
 * decode times, which Matroska does not store and which libavformat derives
 * with its parsers; that part of the cost is on the libavformat side only.
 *
 * Only the benchmark uses this; nothing in Cutlass links FFmpeg. `run.sh`
 * builds it with the system C compiler against the FFmpeg `pkg-config` finds.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavformat/avformat.h>

static int fail(const char *what, int code) {
    char text[128];
    av_strerror(code, text, sizeof text);
    fprintf(stderr, "ffmpeg_matroska: %s: %s\n", what, text);
    return 1;
}

/* The i-th of `count` seek times, as a share of the duration: a fixed
 * shuffle so that the seeks jump back and forth. Same as the example. */
static long long spread(long long duration, long long i, long long count) {
    return duration * ((i * 7919) % count) / count;
}

static int read_all(AVFormatContext *in) {
    AVPacket *packet = av_packet_alloc();
    long long packets = 0, bytes = 0;
    int code;
    while ((code = av_read_frame(in, packet)) >= 0) {
        packets++;
        bytes += packet->size;
        av_packet_unref(packet);
    }
    av_packet_free(&packet);
    if (code != AVERROR_EOF) return fail("read", code);
    printf("packets %lld bytes %lld streams %u\n", packets, bytes, in->nb_streams);
    return 0;
}

static int seek(AVFormatContext *in, long long count) {
    AVStream *stream = in->streams[0];
    long long duration = stream->duration;
    if (duration <= 0 || duration == AV_NOPTS_VALUE)
        duration = av_rescale_q(in->duration, AV_TIME_BASE_Q, stream->time_base);
    AVPacket *packet = av_packet_alloc();
    long long landed = 0, sum = 0;
    for (long long i = 0; i < count; i++) {
        long long at = spread(duration, i, count);
        int code = av_seek_frame(in, 0, at, AVSEEK_FLAG_BACKWARD);
        if (code < 0) return fail("seek", code);
        code = av_read_frame(in, packet);
        if (code >= 0) {
            landed++;
            sum += packet->pts == AV_NOPTS_VALUE ? 0 : packet->pts;
            av_packet_unref(packet);
        }
    }
    av_packet_free(&packet);
    printf("seeks %lld landed %lld pts %lld\n", count, landed, sum);
    return 0;
}

static int remux(AVFormatContext *in, const char *path) {
    AVFormatContext *out = NULL;
    int code = avformat_alloc_output_context2(&out, NULL, NULL, path);
    if (code < 0) return fail("output", code);
    for (unsigned i = 0; i < in->nb_streams; i++) {
        AVStream *to = avformat_new_stream(out, NULL);
        if (!to) return fail("stream", AVERROR(ENOMEM));
        code = avcodec_parameters_copy(to->codecpar, in->streams[i]->codecpar);
        if (code < 0) return fail("parameters", code);
        to->codecpar->codec_tag = 0;
        to->time_base = in->streams[i]->time_base;
    }
    if ((code = avio_open(&out->pb, path, AVIO_FLAG_WRITE)) < 0) return fail("open out", code);
    if ((code = avformat_write_header(out, NULL)) < 0) return fail("header", code);
    AVPacket *packet = av_packet_alloc();
    long long packets = 0;
    while ((code = av_read_frame(in, packet)) >= 0) {
        AVStream *from = in->streams[packet->stream_index];
        AVStream *to = out->streams[packet->stream_index];
        av_packet_rescale_ts(packet, from->time_base, to->time_base);
        packet->pos = -1;
        packets++;
        if ((code = av_interleaved_write_frame(out, packet)) < 0) return fail("write", code);
    }
    if (code != AVERROR_EOF) return fail("read", code);
    if ((code = av_write_trailer(out)) < 0) return fail("trailer", code);
    av_packet_free(&packet);
    avio_closep(&out->pb);
    avformat_free_context(out);
    printf("packets %lld\n", packets);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: ffmpeg_matroska <open|read|seek|remux> <file> [count|out]\n");
        return 2;
    }
    AVFormatContext *in = NULL;
    int code = avformat_open_input(&in, argv[2], NULL, NULL);
    if (code < 0) return fail("open", code);
    int status;
    if (!strcmp(argv[1], "open")) {
        printf("streams %u\n", in->nb_streams);
        status = 0;
    } else if (!strcmp(argv[1], "read")) status = read_all(in);
    else if (!strcmp(argv[1], "seek") && argc == 4) status = seek(in, atoll(argv[3]));
    else if (!strcmp(argv[1], "remux") && argc == 4) {
        if ((code = avformat_find_stream_info(in, NULL)) < 0) return fail("stream info", code);
        status = remux(in, argv[3]);
    }
    else status = 2;
    avformat_close_input(&in);
    return status;
}
