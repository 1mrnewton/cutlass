/*
 * Decodes one file end to end with FFmpeg's libraries and drops the output:
 * the same work `media-bench decode` does with the Rust codecs.
 *
 *     ffmpeg_decode <picture|to:FORMAT|sound|demux> <file> <decoder> <threads> <repeat>
 *
 * libavformat reads the packets of the first picture or sound stream and
 * libavcodec's named decoder turns them into frames. Nothing is resized or
 * kept. `picture` leaves each frame as the decoder made it. `to:FORMAT`
 * (an FFmpeg pixel format name, such as `to:rgb24`) is for a Rust decoder
 * that returns its pictures in another layout than FFmpeg's: libswscale then
 * converts each frame to that layout, so both sides end with the same kind
 * of pixels. `demux` reads every packet of every stream and decodes none,
 * which is what `media-bench demux` does; the decoder name is ignored.
 *
 * The `ffmpeg` command is not used for timing because it also runs a filter
 * graph and an output stage on other threads, which is work the Rust tool
 * does not do.
 *
 * `scripts/bench_media.py` builds this with the system C compiler against
 * the FFmpeg that `pkg-config` finds.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>

static int fail(const char *what, int code) {
    char text[128];
    av_strerror(code, text, sizeof text);
    fprintf(stderr, "ffmpeg_decode: %s: %s\n", what, text);
    return 1;
}

/* Frames are counted and their first sample is read, then dropped. */
static void take(const AVFrame *frame, int sound, long long *count, unsigned *seen) {
    *count += sound ? frame->nb_samples : 1;
    if (frame->data[0]) *seen += frame->data[0][0];
}

/* The picture in the layout the Rust decoder returns. */
static int convert(const AVFrame *frame, enum AVPixelFormat want, unsigned *seen) {
    static struct SwsContext *scale;
    static AVFrame *out;
    if (frame->format == want) return 0;
    scale = sws_getCachedContext(scale, frame->width, frame->height, frame->format, frame->width,
                                 frame->height, want, SWS_BILINEAR, NULL, NULL, NULL);
    if (!scale) return AVERROR(ENOMEM);
    if (!out || out->width != frame->width || out->height != frame->height) {
        av_frame_free(&out);
        out = av_frame_alloc();
        if (!out) return AVERROR(ENOMEM);
        out->format = want;
        out->width = frame->width;
        out->height = frame->height;
        int code = av_frame_get_buffer(out, 0);
        if (code < 0) return code;
    }
    sws_scale(scale, (const uint8_t *const *)frame->data, frame->linesize, 0, frame->height,
              out->data, out->linesize);
    *seen += out->data[0][0];
    return 0;
}

/* Every packet of every stream, read and dropped. */
static int packets(const char *path, long long *count, long long *bytes) {
    AVFormatContext *file = NULL;
    int code = avformat_open_input(&file, path, NULL, NULL);
    if (code < 0) return fail("open", code);
    code = avformat_find_stream_info(file, NULL);
    if (code < 0) return fail("stream info", code);
    AVPacket *packet = av_packet_alloc();
    unsigned seen = 0;
    for (;;) {
        code = av_read_frame(file, packet);
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

static int once(const char *path, const char *name, int sound, enum AVPixelFormat want,
                int threads, long long *count) {
    AVFormatContext *file = NULL;
    int code = avformat_open_input(&file, path, NULL, NULL);
    if (code < 0) return fail("open", code);
    code = avformat_find_stream_info(file, NULL);
    if (code < 0) return fail("stream info", code);
    enum AVMediaType type = sound ? AVMEDIA_TYPE_AUDIO : AVMEDIA_TYPE_VIDEO;
    int stream = av_find_best_stream(file, type, -1, -1, NULL, 0);
    if (stream < 0) return fail("no such stream", stream);
    const AVCodec *codec = avcodec_find_decoder_by_name(name);
    if (!codec) {
        fprintf(stderr, "ffmpeg_decode: no decoder named %s\n", name);
        return 1;
    }
    AVCodecContext *decoder = avcodec_alloc_context3(codec);
    code = avcodec_parameters_to_context(decoder, file->streams[stream]->codecpar);
    if (code < 0) return fail("parameters", code);
    decoder->thread_count = threads;
    code = avcodec_open2(decoder, codec, NULL);
    if (code < 0) return fail("open decoder", code);

    AVPacket *packet = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    unsigned seen = 0;
    int end = 0;
    while (!end) {
        code = av_read_frame(file, packet);
        if (code == AVERROR_EOF) {
            end = 1;
            code = avcodec_send_packet(decoder, NULL);
        } else if (code < 0) {
            return fail("read", code);
        } else if (packet->stream_index != stream) {
            av_packet_unref(packet);
            continue;
        } else {
            code = avcodec_send_packet(decoder, packet);
            av_packet_unref(packet);
        }
        if (code < 0) return fail("send", code);
        for (;;) {
            code = avcodec_receive_frame(decoder, frame);
            if (code == AVERROR(EAGAIN) || code == AVERROR_EOF) break;
            if (code < 0) return fail("receive", code);
            take(frame, sound, count, &seen);
            if (want != AV_PIX_FMT_NONE) {
                code = convert(frame, want, &seen);
                if (code < 0) return fail("convert", code);
            }
            av_frame_unref(frame);
        }
    }
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&decoder);
    avformat_close_input(&file);
    return seen == 0xffffffffu ? 2 : 0;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: ffmpeg_decode <picture|to:FORMAT|sound|demux> <file> <decoder> "
                        "<threads> <repeat>\n");
        return 1;
    }
    int sound = strcmp(argv[1], "sound") == 0;
    enum AVPixelFormat want = AV_PIX_FMT_NONE;
    if (strncmp(argv[1], "to:", 3) == 0) {
        want = av_get_pix_fmt(argv[1] + 3);
        if (want == AV_PIX_FMT_NONE) {
            fprintf(stderr, "ffmpeg_decode: no pixel format named %s\n", argv[1] + 3);
            return 1;
        }
    }
    int threads = atoi(argv[4]);
    int repeat = atoi(argv[5]);
    long long count = 0;
    av_log_set_level(AV_LOG_ERROR);
    if (strcmp(argv[1], "demux") == 0) {
        long long bytes = 0;
        for (int turn = 0; turn < repeat; turn++) {
            count = 0;
            bytes = 0;
            int code = packets(argv[2], &count, &bytes);
            if (code) return code;
        }
        printf("packets %lld bytes %lld\n", count, bytes);
        return 0;
    }
    for (int turn = 0; turn < repeat; turn++) {
        count = 0;
        int code = once(argv[2], argv[3], sound, want, threads, &count);
        if (code) return code;
    }
    printf("%s %lld\n", sound ? "sound frames" : "pictures", count);
    return 0;
}
