/* Builds one IVF whose inter frames carry segment features. libvpx applies
   this map only in realtime mode at speed 5 or higher, and only on an inter
   frame. The default map is a loop-filter step and a reference. Passing
   "skip" marks segment 1 as skip instead. Segment 3 is left alone: libvpx
   withholds that id for 20 frames after a key. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vpx/vp8cx.h>
#include <vpx/vpx_encoder.h>

static void die(vpx_codec_ctx_t *ctx, const char *what) {
    fprintf(stderr, "%s: %s\n", what, vpx_codec_error(ctx));
    exit(1);
}

static void fill(vpx_image_t *img, int w, int h, int seed) {
    int y, x, cw, ch;
    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            img->planes[0][y * img->stride[0] + x] =
                (unsigned char)((x * 13 + y * 7 + seed * 17) & 255);
        }
    }
    cw = w / 2;
    ch = h / 2;
    for (y = 0; y < ch; y++) {
        for (x = 0; x < cw; x++) {
            img->planes[1][y * img->stride[1] + x] = (unsigned char)(30 + (x + seed) % 90);
            img->planes[2][y * img->stride[2] + x] = (unsigned char)(140 + (y * 3 + seed) % 70);
        }
    }
}

static void write_ivf(const char *path, int w, int h, unsigned char **pkts, size_t *sizes,
                      int count) {
    FILE *f = fopen(path, "wb");
    unsigned char hdr[32];
    int i;
    if (!f) exit(1);
    memset(hdr, 0, sizeof(hdr));
    memcpy(hdr, "DKIF", 4);
    hdr[6] = 32;
    memcpy(hdr + 8, "VP90", 4);
    hdr[12] = (unsigned char)(w & 255);
    hdr[13] = (unsigned char)((w >> 8) & 255);
    hdr[14] = (unsigned char)(h & 255);
    hdr[15] = (unsigned char)((h >> 8) & 255);
    hdr[16] = 30;
    hdr[20] = 1;
    hdr[24] = (unsigned char)count;
    fwrite(hdr, 1, 32, f);
    for (i = 0; i < count; i++) {
        unsigned char fh[12];
        unsigned int sz = (unsigned int)sizes[i];
        unsigned long long pts = (unsigned long long)i;
        memset(fh, 0, sizeof(fh));
        memcpy(fh, &sz, 4);
        memcpy(fh + 4, &pts, 8);
        fwrite(fh, 1, 12, f);
        fwrite(pkts[i], 1, sizes[i], f);
    }
    fclose(f);
}

int main(int argc, char **argv) {
    const int w = 160;
    const int h = 96;
    const int frames = 4;
    vpx_codec_ctx_t codec;
    vpx_codec_enc_cfg_t cfg;
    vpx_codec_iface_t *iface = vpx_codec_vp9_cx();
    int cols, rows, n, count, y, x, skip_only;
    unsigned char *map;
    vpx_roi_map_t roi;
    vpx_image_t img;
    unsigned char *stored[4];
    size_t stored_sz[4];
    skip_only = argc == 3 && strcmp(argv[2], "skip") == 0;
    if (argc != 2 && !skip_only) return 1;
    if (vpx_codec_enc_config_default(iface, &cfg, 0)) return 1;
    cfg.g_w = (unsigned)w;
    cfg.g_h = (unsigned)h;
    cfg.g_timebase.num = 1;
    cfg.g_timebase.den = 30;
    cfg.g_threads = 1;
    cfg.g_lag_in_frames = 0;
    cfg.rc_end_usage = VPX_Q;
    cfg.rc_min_quantizer = 10;
    cfg.rc_max_quantizer = 48;
    cfg.kf_min_dist = 0;
    cfg.kf_max_dist = 999;
    if (vpx_codec_enc_init(&codec, iface, &cfg, 0)) die(&codec, "init");
    if (vpx_codec_control(&codec, VP8E_SET_CPUUSED, 8)) die(&codec, "cpu");
    if (vpx_codec_control(&codec, VP8E_SET_CQ_LEVEL, 40)) die(&codec, "cq");

    cols = w / 8;
    rows = h / 8;
    map = calloc((size_t)cols * (size_t)rows, 1);
    if (!map) return 1;
    for (y = 0; y < rows; y++) {
        for (x = 0; x < cols; x++) map[y * cols + x] = (x < cols / 2) ? 1 : 2;
    }
    memset(&roi, 0, sizeof(roi));
    roi.enabled = 1;
    roi.roi_map = map;
    roi.rows = (unsigned)rows;
    roi.cols = (unsigned)cols;
    for (n = 0; n < 8; n++) roi.ref_frame[n] = -1;
    if (skip_only) {
        roi.skip[1] = 1;
    } else {
        roi.delta_lf[1] = -63;
        roi.ref_frame[2] = 1;
    }
    if (vpx_codec_control(&codec, VP9E_SET_ROI_MAP, &roi)) die(&codec, "roi");

    if (!vpx_img_alloc(&img, VPX_IMG_FMT_I420, (unsigned)w, (unsigned)h, 1)) return 1;
    count = 0;
    for (n = 0; n < frames; n++) {
        vpx_codec_iter_t iter = NULL;
        const vpx_codec_cx_pkt_t *pkt;
        vpx_enc_frame_flags_t flags = n == 0 ? VPX_EFLAG_FORCE_KF : 0;
        fill(&img, w, h, n + 1);
        if (vpx_codec_encode(&codec, &img, n, 1, flags, VPX_DL_REALTIME)) die(&codec, "enc");
        while ((pkt = vpx_codec_get_cx_data(&codec, &iter))) {
            if (pkt->kind != VPX_CODEC_CX_FRAME_PKT || count >= frames) continue;
            stored[count] = malloc(pkt->data.frame.sz);
            if (!stored[count]) return 1;
            memcpy(stored[count], pkt->data.frame.buf, pkt->data.frame.sz);
            stored_sz[count] = pkt->data.frame.sz;
            count++;
        }
    }
    write_ivf(argv[1], w, h, stored, stored_sz, count);
    vpx_img_free(&img);
    vpx_codec_destroy(&codec);
    free(map);
    return count == frames ? 0 : 1;
}
